// ---------------------------------------------------------------------------
// tracker.rs — Greedy IoU multi-object tracker
//
// Provides frame-to-frame identity tracking for vehicles and other detections.
// Used by the live stream path (not the stateless /execute controller).
// ---------------------------------------------------------------------------

use crate::alpen_detect::ALPEN_DETECT_ACCIDENT_CLASSES;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration (overridable via env vars)
// ---------------------------------------------------------------------------

/// Minimum IoU between a detection and an existing track to be considered a match.
fn iou_match_threshold() -> f32 {
    std::env::var("ALPEN_IOU_MATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.3)
}

/// Number of consecutive frames a track can go unseen before being removed.
fn track_ttl_frames() -> u64 {
    std::env::var("ALPEN_TRACK_TTL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// EMA decay factor for per-track confidence accumulation.
fn conf_decay() -> f32 {
    std::env::var("ALPEN_CONF_DECAY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.85)
}

/// Accumulated confidence threshold for emitting a tracker-based accident alert.
fn conf_accum_threshold() -> f32 {
    std::env::var("ALPEN_CONF_ACCUM_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5)
}

/// Minimum number of frames a track must have been seen before its accumulated
/// confidence can trigger an alert.
const MIN_FRAMES_FOR_ALERT: u64 = 2;

/// Maximum number of confidence values stored per track.
const CONF_HISTORY_CAP: usize = 10;

/// Dwell time threshold (seconds) before a track is considered stalled.
fn stall_threshold_sec() -> f64 {
    std::env::var("ALPEN_STALL_THRESHOLD_SEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(90.0)
}

/// Maximum pixel movement for a track to be considered "stationary".
const STALL_PIXEL_THRESHOLD: f32 = 5.0;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Track {
    pub id: u32,
    pub bbox: [f32; 4],
    pub class: u8,
    pub entered_at_sec: f64,
    pub last_seen_frame: u64,
    pub last_seen_sec: f64,
    pub frames_seen: u64,
    pub roi_membership: HashMap<String, bool>,
    pub conf_history: VecDeque<f32>,
    pub accum_conf: f32,
    /// Center position when the track was first created or last moved significantly.
    pub anchor_center: [f32; 2],
    /// Current center (updated every matched frame).
    pub last_center: [f32; 2],
    /// Whether a stalled event has already been emitted for this occupancy period.
    pub stalled_emitted: bool,
}

#[derive(Clone, Debug)]
pub enum TrackEvent {
    Entered {
        track_id: u32,
        bbox: [f32; 4],
    },
    Exited {
        track_id: u32,
        dwell_sec: f32,
        bbox: [f32; 4],
    },
    Stalled {
        track_id: u32,
        dwell_sec: f32,
        bbox: [f32; 4],
    },
}

pub struct Tracker {
    tracks: HashMap<u32, Track>,
    next_id: u32,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
        }
    }

    /// Update tracker with new detections. Returns events (entered, exited, stalled).
    pub fn update(
        &mut self,
        detections: &[[f32; 6]],
        frame_idx: u64,
        timestamp_sec: f64,
    ) -> Vec<TrackEvent> {
        let iou_thresh = iou_match_threshold();
        let ttl = track_ttl_frames();
        let decay = conf_decay();
        let stall_sec = stall_threshold_sec();

        let mut events = Vec::new();

        // Step 1: compute IoU between all detections and all active tracks
        let track_ids: Vec<u32> = self.tracks.keys().copied().collect();
        let mut pairs: Vec<(usize, u32, f32)> = Vec::new(); // (det_idx, track_id, iou)

        for (di, det) in detections.iter().enumerate() {
            for &tid in &track_ids {
                if let Some(track) = self.tracks.get(&tid) {
                    let iou_val = bbox_iou(det, &track.bbox);
                    if iou_val >= iou_thresh {
                        pairs.push((di, tid, iou_val));
                    }
                }
            }
        }

        // Step 2: greedy assignment sorted by IoU descending
        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut det_matched = vec![false; detections.len()];
        let mut track_matched: HashMap<u32, bool> = HashMap::new();

        for &(di, tid, _) in &pairs {
            if det_matched[di] || *track_matched.get(&tid).unwrap_or(&false) {
                continue;
            }
            det_matched[di] = true;
            track_matched.insert(tid, true);

            // Update the matched track
            if let Some(track) = self.tracks.get_mut(&tid) {
                let det = &detections[di];
                track.bbox = [det[0], det[1], det[2], det[3]];
                track.class = det[5] as u8;
                track.last_seen_frame = frame_idx;
                track.last_seen_sec = timestamp_sec;
                track.frames_seen += 1;
                let cx = (det[0] + det[2]) * 0.5;
                let cy = (det[1] + det[3]) * 0.5;
                track.last_center = [cx, cy];

                // Check if the track moved significantly — reset anchor if so
                let dx = cx - track.anchor_center[0];
                let dy = cy - track.anchor_center[1];
                if (dx * dx + dy * dy).sqrt() > STALL_PIXEL_THRESHOLD {
                    track.anchor_center = [cx, cy];
                    track.stalled_emitted = false;
                }

                // Confidence accumulation for accident classes
                let class_id = det[5] as usize;
                if ALPEN_DETECT_ACCIDENT_CLASSES.contains(&class_id) {
                    track.accum_conf = track.accum_conf * decay + det[4];
                    if track.conf_history.len() >= CONF_HISTORY_CAP {
                        track.conf_history.pop_front();
                    }
                    track.conf_history.push_back(det[4]);
                }

                // Stalled detection
                let dwell = (timestamp_sec - track.entered_at_sec) as f32;
                if dwell as f64 > stall_sec && !track.stalled_emitted {
                    let move_dx = cx - track.anchor_center[0];
                    let move_dy = cy - track.anchor_center[1];
                    if (move_dx * move_dx + move_dy * move_dy).sqrt() <= STALL_PIXEL_THRESHOLD {
                        track.stalled_emitted = true;
                        events.push(TrackEvent::Stalled {
                            track_id: tid,
                            dwell_sec: dwell,
                            bbox: track.bbox,
                        });
                    }
                }
            }
        }

        // Step 3: create new tracks for unmatched detections
        for (di, det) in detections.iter().enumerate() {
            if det_matched[di] {
                continue;
            }
            let cx = (det[0] + det[2]) * 0.5;
            let cy = (det[1] + det[3]) * 0.5;
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);

            let mut conf_hist = VecDeque::with_capacity(CONF_HISTORY_CAP);
            let mut accum = 0.0f32;
            let class_id = det[5] as usize;
            if ALPEN_DETECT_ACCIDENT_CLASSES.contains(&class_id) {
                conf_hist.push_back(det[4]);
                accum = det[4];
            }

            self.tracks.insert(
                id,
                Track {
                    id,
                    bbox: [det[0], det[1], det[2], det[3]],
                    class: det[5] as u8,
                    entered_at_sec: timestamp_sec,
                    last_seen_frame: frame_idx,
                    last_seen_sec: timestamp_sec,
                    frames_seen: 1,
                    roi_membership: HashMap::new(),
                    conf_history: conf_hist,
                    accum_conf: accum,
                    anchor_center: [cx, cy],
                    last_center: [cx, cy],
                    stalled_emitted: false,
                },
            );
            events.push(TrackEvent::Entered {
                track_id: id,
                bbox: [det[0], det[1], det[2], det[3]],
            });
        }

        // Step 4: age out stale tracks
        let stale: Vec<u32> = self
            .tracks
            .iter()
            .filter(|(_, t)| frame_idx.saturating_sub(t.last_seen_frame) > ttl)
            .map(|(&id, _)| id)
            .collect();

        for id in stale {
            if let Some(track) = self.tracks.remove(&id) {
                let dwell = (track.last_seen_sec - track.entered_at_sec) as f32;
                events.push(TrackEvent::Exited {
                    track_id: id,
                    dwell_sec: dwell,
                    bbox: track.bbox,
                });
            }
        }

        events
    }

    /// Coast: age tracks by one frame without any detections (used when YOLO is skipped).
    pub fn coast(&mut self, frame_idx: u64, timestamp_sec: f64) -> Vec<TrackEvent> {
        let ttl = track_ttl_frames();
        let mut events = Vec::new();

        let stale: Vec<u32> = self
            .tracks
            .iter()
            .filter(|(_, t)| frame_idx.saturating_sub(t.last_seen_frame) > ttl)
            .map(|(&id, _)| id)
            .collect();

        for id in stale {
            if let Some(track) = self.tracks.remove(&id) {
                let dwell = (track.last_seen_sec - track.entered_at_sec) as f32;
                events.push(TrackEvent::Exited {
                    track_id: id,
                    dwell_sec: dwell,
                    bbox: track.bbox,
                });
            }
        }

        let _ = timestamp_sec; // used for stall checks in future if needed
        events
    }

    /// Returns tracks whose accumulated confidence exceeds the threshold and
    /// have been seen for at least `MIN_FRAMES_FOR_ALERT` frames.
    /// Returns: Vec<(track_id, accum_conf, bbox)>
    pub fn accumulated_alerts(&self) -> Vec<(u32, f32, [f32; 4])> {
        let thresh = conf_accum_threshold();
        self.tracks
            .values()
            .filter(|t| t.accum_conf >= thresh && t.frames_seen >= MIN_FRAMES_FOR_ALERT)
            .map(|t| (t.id, t.accum_conf, t.bbox))
            .collect()
    }

    pub fn active_tracks(&self) -> &HashMap<u32, Track> {
        &self.tracks
    }

    pub fn active_track_count(&self) -> usize {
        self.tracks.len()
    }
}

// ---------------------------------------------------------------------------
// IoU helper
// ---------------------------------------------------------------------------

fn bbox_iou(det: &[f32; 6], bbox: &[f32; 4]) -> f32 {
    let x1 = det[0].max(bbox[0]);
    let y1 = det[1].max(bbox[1]);
    let x2 = det[2].min(bbox[2]);
    let y2 = det[3].min(bbox[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let intersection = inter_w * inter_h;

    let area_a = (det[2] - det[0]) * (det[3] - det[1]);
    let area_b = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        return 0.0;
    }

    intersection / union
}
