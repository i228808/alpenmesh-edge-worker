use crate::alpen_detect::{
    ALPEN_DETECT_NUM_ANCHORS, ALPEN_DETECT_NUM_CLASSES, ALPEN_DETECT_VEHICLE_CLASSES,
    confidence_threshold_for_class,
};
use serde::Serialize;
use std::collections::HashMap;

pub type RoiMap = HashMap<String, Vec<[i32; 2]>>;

#[derive(Clone, Debug, Serialize)]
pub struct TrafficLightMetric {
    pub color: String,
    pub nearest_roi: String,
    pub confidence: f32,
}

pub fn decode_output(raw_data: &[f32]) -> Vec<[f32; 6]> {
    let expected = (4 + ALPEN_DETECT_NUM_CLASSES) * ALPEN_DETECT_NUM_ANCHORS;
    if raw_data.len() < expected {
        return Vec::new();
    }

    let mut results = Vec::with_capacity(128);
    for anchor in 0..ALPEN_DETECT_NUM_ANCHORS {
        let mut best_score = 0.0f32;
        let mut best_class = None;

        for class_id in 0..ALPEN_DETECT_NUM_CLASSES {
            let score_idx = (4 + class_id) * ALPEN_DETECT_NUM_ANCHORS + anchor;
            let score = raw_data[score_idx];
            if score > best_score {
                best_score = score;
                best_class = Some(class_id);
            }
        }

        let Some(class_id) = best_class else {
            continue;
        };
        if best_score < confidence_threshold_for_class(class_id) {
            continue;
        }

        let cx = raw_data[anchor];
        let cy = raw_data[ALPEN_DETECT_NUM_ANCHORS + anchor];
        let w = raw_data[2 * ALPEN_DETECT_NUM_ANCHORS + anchor];
        let h = raw_data[3 * ALPEN_DETECT_NUM_ANCHORS + anchor];
        let x1 = (cx - w / 2.0).clamp(0.0, 640.0);
        let y1 = (cy - h / 2.0).clamp(0.0, 640.0);
        let x2 = (cx + w / 2.0).clamp(0.0, 640.0);
        let y2 = (cy + h / 2.0).clamp(0.0, 640.0);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        results.push([x1, y1, x2, y2, best_score, class_id as f32]);
    }

    results
}

pub fn compute_roi_vehicle_counts(
    detections: &[[f32; 6]],
    rois: &RoiMap,
    original_w: u32,
    original_h: u32,
) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = rois.keys().map(|k| (k.clone(), 0)).collect();
    if rois.is_empty() {
        return counts;
    }

    let mut roi_keys: Vec<String> = rois.keys().cloned().collect();
    roi_keys.sort();

    for det in detections {
        if !is_vehicle_class(det[5] as i32) {
            continue;
        }

        let cx = ((det[0] + det[2]) * 0.5) * original_w as f32 / 640.0;
        let cy = ((det[1] + det[3]) * 0.5) * original_h as f32 / 640.0;

        for key in &roi_keys {
            let poly = match rois.get(key) {
                Some(p) => p,
                None => continue,
            };
            if point_in_polygon(cx, cy, poly) {
                if let Some(v) = counts.get_mut(key) {
                    *v += 1;
                }
                break;
            }
        }
    }

    counts
}

pub fn compute_congestion_levels(counts: &HashMap<String, u32>) -> HashMap<String, String> {
    counts
        .iter()
        .map(|(k, v)| {
            let level = if *v >= 10 {
                "High"
            } else if *v >= 5 {
                "Medium"
            } else {
                "Low"
            };
            (k.clone(), level.to_string())
        })
        .collect()
}

pub fn detect_traffic_lights_from_detections(
    frame: &image::RgbImage,
    detections: &[[f32; 6]],
    rois: &RoiMap,
    original_w: u32,
    original_h: u32,
) -> Vec<TrafficLightMetric> {
    let (w, h) = frame.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut outputs = Vec::new();

    for det in detections {
        if !is_traffic_light_class(det[5] as i32) {
            continue;
        }

        let x1 = ((det[0] * original_w as f32) / 640.0).floor().max(0.0) as u32;
        let y1 = ((det[1] * original_h as f32) / 640.0).floor().max(0.0) as u32;
        let x2 = ((det[2] * original_w as f32) / 640.0).ceil().min(w as f32) as u32;
        let y2 = ((det[3] * original_h as f32) / 640.0).ceil().min(h as f32) as u32;

        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        let mut red_pixels: u32 = 0;
        let mut yellow_pixels: u32 = 0;
        let mut green_pixels: u32 = 0;

        for y in y1..y2 {
            for x in x1..x2 {
                let p = frame.get_pixel(x, y).0;
                let (hue, sat, val) = rgb_to_hsv(p[0], p[1], p[2]);

                if val > 0.39 && sat > 0.39 {
                    if !(15.0..=345.0).contains(&hue) {
                        red_pixels += 1;
                    } else if (15.0..=35.0).contains(&hue) {
                        yellow_pixels += 1;
                    } else if (40.0..=90.0).contains(&hue) {
                        green_pixels += 1;
                    }
                }
            }
        }

        let pixel_counts = [
            ("Red", red_pixels),
            ("Yellow", yellow_pixels),
            ("Green", green_pixels),
        ];

        let (best_color, best_val) = match pixel_counts.into_iter().max_by_key(|(_, v)| *v) {
            Some(v) => v,
            None => continue,
        };

        let color = if best_val > 20 {
            best_color.to_string()
        } else {
            "Off".to_string()
        };

        let cx = (x1 + x2) as f32 * 0.5;
        let cy = (y1 + y2) as f32 * 0.5;
        let nearest_roi = nearest_roi_label(cx, cy, rois).unwrap_or_else(|| "None".to_string());

        outputs.push(TrafficLightMetric {
            color,
            nearest_roi,
            confidence: det[4],
        });
    }

    outputs
}

fn is_vehicle_class(class_id: i32) -> bool {
    ALPEN_DETECT_VEHICLE_CLASSES.contains(&(class_id as usize))
}

fn is_traffic_light_class(class_id: i32) -> bool {
    let _ = class_id;
    false
}

fn nearest_roi_label(x: f32, y: f32, rois: &RoiMap) -> Option<String> {
    if rois.is_empty() {
        return None;
    }

    for (label, poly) in rois {
        if point_in_polygon(x, y, poly) {
            return Some(label.clone());
        }
    }

    let mut best_label: Option<String> = None;
    let mut best_dist = f32::MAX;
    for (label, poly) in rois {
        if poly.is_empty() {
            continue;
        }
        let (cx, cy) = polygon_centroid(poly);
        let dx = cx - x;
        let dy = cy - y;
        let dist = dx * dx + dy * dy;
        if dist < best_dist {
            best_dist = dist;
            best_label = Some(label.clone());
        }
    }

    best_label
}

fn polygon_centroid(poly: &[[i32; 2]]) -> (f32, f32) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let n = poly.len().max(1) as f32;
    for p in poly {
        sx += p[0] as f32;
        sy += p[1] as f32;
    }
    (sx / n, sy / n)
}

fn point_in_polygon(x: f32, y: f32, poly: &[[i32; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let xi = poly[i][0] as f32;
        let yi = poly[i][1] as f32;
        let xj = poly[j][0] as f32;
        let yj = poly[j][1] as f32;

        let intersects = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / ((yj - yi).abs().max(f32::EPSILON)) + xi);

        if intersects {
            inside = !inside;
        }
        j = i;
    }

    inside
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let c_max = rf.max(gf).max(bf);
    let c_min = rf.min(gf).min(bf);
    let delta = c_max - c_min;

    let hue = if delta <= f32::EPSILON {
        0.0
    } else if (c_max - rf).abs() <= f32::EPSILON {
        60.0 * ((gf - bf) / delta).rem_euclid(6.0)
    } else if (c_max - gf).abs() <= f32::EPSILON {
        60.0 * (((bf - rf) / delta) + 2.0)
    } else {
        60.0 * (((rf - gf) / delta) + 4.0)
    };

    let sat = if c_max <= f32::EPSILON {
        0.0
    } else {
        delta / c_max
    };
    (hue, sat, c_max)
}

pub fn apply_nms(mut boxes: Vec<[f32; 6]>, iou_threshold: f32) -> Vec<[f32; 6]> {
    // Sort by score descending
    boxes.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }

        kept.push(boxes[i]);

        for j in (i + 1)..boxes.len() {
            if suppressed[j] {
                continue;
            }

            if iou(&boxes[i], &boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}

fn iou(box_a: &[f32; 6], box_b: &[f32; 6]) -> f32 {
    let x1 = box_a[0].max(box_b[0]);
    let y1 = box_a[1].max(box_b[1]);
    let x2 = box_a[2].min(box_b[2]);
    let y2 = box_a[3].min(box_b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let intersection = inter_w * inter_h;

    let area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    let area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);

    if area_a + area_b - intersection <= 0.0 {
        return 0.0;
    }

    intersection / (area_a + area_b - intersection)
}

// ---------------------------------------------------------------------------
// Phase 7: Tracker-based ROI traffic metrics
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct RoiTrafficMetrics {
    /// Current number of tracked vehicles in this ROI.
    pub occupancy: u32,
    /// Mean dwell time (seconds) of vehicles currently in the ROI.
    pub mean_dwell_sec: f32,
    /// Number of tracks in this ROI flagged as stalled.
    pub stalled_count: u32,
    /// Vehicles that exited the ROI with dwell between 8–30s in the recent window.
    pub slow_throughput: u32,
    /// Rate of occupancy change: (current - previous) / window_sec.
    pub queue_growth_rate: f32,
}

/// Compute per-ROI traffic metrics from the tracker state.
///
/// This replaces `compute_roi_vehicle_counts` for the live stream path. It
/// provides richer signals including dwell time, stall counts, slow throughput,
/// and queue growth rate.
///
/// `previous_occupancy` is the occupancy map from the prior window — if `None`,
/// queue_growth_rate will be 0.
pub fn compute_roi_traffic_metrics(
    tracker: &crate::tracker::Tracker,
    rois: &RoiMap,
    img_w: u32,
    img_h: u32,
    current_sec: f64,
    previous_occupancy: Option<&HashMap<String, u32>>,
    window_sec: f32,
) -> HashMap<String, RoiTrafficMetrics> {
    let mut metrics: HashMap<String, RoiTrafficMetrics> = rois
        .keys()
        .map(|k| {
            (
                k.clone(),
                RoiTrafficMetrics {
                    occupancy: 0,
                    mean_dwell_sec: 0.0,
                    stalled_count: 0,
                    slow_throughput: 0,
                    queue_growth_rate: 0.0,
                },
            )
        })
        .collect();

    if rois.is_empty() {
        return metrics;
    }

    // For each active track, determine which ROI it belongs to
    for track in tracker.active_tracks().values() {
        let cx = (track.bbox[0] + track.bbox[2]) * 0.5 * img_w as f32 / 640.0;
        let cy = (track.bbox[1] + track.bbox[3]) * 0.5 * img_h as f32 / 640.0;

        for (roi_name, poly) in rois {
            if point_in_polygon(cx, cy, poly) {
                if let Some(m) = metrics.get_mut(roi_name) {
                    let dwell = (current_sec - track.entered_at_sec) as f32;
                    m.occupancy += 1;
                    m.mean_dwell_sec += dwell;

                    if track.stalled_emitted {
                        m.stalled_count += 1;
                    }

                    // Slow throughput: vehicles dwelling 8–30s (recently exited
                    // would be ideal, but as an approximation we count active
                    // tracks in that range too)
                    if dwell >= 8.0 && dwell <= 30.0 {
                        m.slow_throughput += 1;
                    }
                }
                break; // each track in at most one ROI
            }
        }
    }

    // Finalize mean dwell and queue growth
    for (name, m) in metrics.iter_mut() {
        if m.occupancy > 0 {
            m.mean_dwell_sec /= m.occupancy as f32;
        }
        if let Some(prev) = previous_occupancy {
            let prev_occ = prev.get(name).copied().unwrap_or(0);
            if window_sec > 0.0 {
                m.queue_growth_rate =
                    (m.occupancy as f32 - prev_occ as f32) / window_sec;
            }
        }
    }

    metrics
}
