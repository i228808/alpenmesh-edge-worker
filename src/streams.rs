use crate::alpen_detect::{self, AccidentReport, DetectionSummary};
use crate::state::{PerStreamState, WorkerState};
use crate::stalled::StalledVehicleReport;
use crate::tracker::{TrackEvent, Tracker};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Semaphore;
use tracing::{error, info};

static CONNECT_LIMITER: OnceLock<Arc<Semaphore>> = OnceLock::new();

pub struct StreamManager {
    pub state: WorkerState,
}

impl StreamManager {
    pub fn new(state: WorkerState) -> Self {
        Self { state }
    }

    pub async fn spawn_stream(&self, stream_id: String, url: String) {
        let state = self.state.clone();
        let metrics_client = reqwest::Client::new();
        let scheduler_client = reqwest::Client::new();
        let limiter = CONNECT_LIMITER
            .get_or_init(|| Arc::new(Semaphore::new(2)))
            .clone();
        
        tokio::spawn(async move {
            info!("Spawning Native Ingestion for stream {}: {}", stream_id, url);
            state.active_streams.fetch_add(1, Ordering::SeqCst);

            let final_url = normalize_stream_url(&url);
            let mut consecutive_failures: u32 = 0;

            // Ensure per-stream state exists
            let stream_state = {
                let need_insert = {
                    let map = state.per_stream.read().unwrap();
                    !map.contains_key(&stream_id)
                };
                if need_insert {
                    let mut map = state.per_stream.write().unwrap();
                    map.entry(stream_id.clone())
                        .or_insert_with(|| {
                            Arc::new(Mutex::new(PerStreamState {
                                tracker: Tracker::new(),
                                last_roi_hashes: HashMap::new(),
                                last_frame_full_hash: None,
                                congestion_cooldown_until: 0.0,
                                last_frame_idx: 0,
                            }))
                        });
                }
                let map = state.per_stream.read().unwrap();
                map.get(&stream_id).unwrap().clone()
            };

            let mut frame_idx: u64 = 0;

            loop {
                let _permit = match limiter.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => break,
                };

                let mut args = vec![];

                if final_url.starts_with("rtsp://") {
                    args.push("-rtsp_transport");
                    args.push("tcp");
                }

                if final_url.starts_with("http://") || final_url.starts_with("https://") {
                    args.extend_from_slice(&[
                        "-user_agent",
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        "-rw_timeout",
                        "15000000",
                        "-reconnect",
                        "1",
                        "-reconnect_streamed",
                        "1",
                        "-reconnect_on_network_error",
                        "1",
                        "-reconnect_on_http_error",
                        "4xx,5xx",
                        "-reconnect_delay_max",
                        "10",
                        "-http_persistent",
                        "1",
                    ]);
                }

                args.extend_from_slice(&[
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-hwaccel",
                    "nvdec",
                    "-i",
                    &final_url,
                    "-vf",
                    "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:color=black",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-r",
                    "10",
                    "-an",
                    "-",
                ]);

                let child_spawn = Command::new("ffmpeg")
                    .args(&args)
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::inherit())
                    .spawn();

                let mut child: Child = match child_spawn {
                    Ok(c) => c,
                    Err(e) => {
                        consecutive_failures += 1;
                        error!("Failed to spawn ffmpeg for {}: {}", stream_id, e);
                        sleep_with_backoff(&stream_id, consecutive_failures).await;
                        continue;
                    }
                };

                drop(_permit);

                let mut stdout = match child.stdout.take() {
                    Some(s) => s,
                    None => {
                        consecutive_failures += 1;
                        error!("Failed to open ffmpeg stdout for {}", stream_id);
                        let _ = child.kill().await;
                        let _ = child.wait().await;
                        sleep_with_backoff(&stream_id, consecutive_failures).await;
                        continue;
                    }
                };

                let frame_size = 640 * 640 * 3;
                let mut buffer = vec![0u8; frame_size];
                let mut got_any_frame = false;

                loop {
                    match stdout.read_exact(&mut buffer).await {
                        Ok(_) => {
                            got_any_frame = true;
                            consecutive_failures = 0;
                            frame_idx += 1;
                            let rgb_data = buffer.clone();
                            let state_inner = state.clone();
                            let sid = stream_id.clone();
                            let client_inner = metrics_client.clone();
                            let ss = stream_state.clone();
                            let fidx = frame_idx;

                            tokio::spawn(async move {
                                let timestamp = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_secs_f64())
                                    .unwrap_or(0.0);

                                let stream_rois = {
                                    let rois = state_inner.stream_rois.read().unwrap();
                                    rois.get(&sid).cloned().unwrap_or_default()
                                };

                                // --- Phase 9: Motion pre-filter ---
                                let (should_skip, new_roi_hashes, new_full_hash) = {
                                    let pss = ss.lock().unwrap();
                                    crate::motion::should_skip_frame(
                                        &rgb_data,
                                        &stream_rois,
                                        &pss.last_roi_hashes,
                                        pss.last_frame_full_hash.as_ref(),
                                    )
                                };

                                // Update hashes in per-stream state
                                {
                                    let mut pss = ss.lock().unwrap();
                                    pss.last_roi_hashes = new_roi_hashes;
                                    pss.last_frame_full_hash = Some(new_full_hash);
                                    pss.last_frame_idx = fidx;
                                }

                                if should_skip {
                                    // Coast the tracker (age tracks without new detections)
                                    let events = {
                                        let mut pss = ss.lock().unwrap();
                                        pss.tracker.coast(fidx, timestamp)
                                    };
                                    handle_track_events_stalled(
                                        &events,
                                        &sid,
                                        timestamp,
                                        &client_inner,
                                        &state_inner,
                                    )
                                    .await;
                                    return; // skip rest of processing
                                }

                                let frame_for_metrics = rgb_data.clone();
                                let detection_summary = detect_with_optional_enhancement(
                                    &state_inner,
                                    &frame_for_metrics,
                                    timestamp,
                                );
                                let detections = detection_summary.detections.clone();
                                state_inner.record_inference(&sid);

                                // --- Phase 5: Update tracker ---
                                let (track_events, track_count, accumulated_alerts) = {
                                    let mut pss = ss.lock().unwrap();
                                    let events =
                                        pss.tracker.update(&detections, fidx, timestamp);
                                    let count = pss.tracker.active_track_count();
                                    let alerts = pss.tracker.accumulated_alerts();
                                    (events, count, alerts)
                                };

                                // --- Phase 6: Generate AccidentReports from accumulated alerts ---
                                let mut all_reports = detection_summary.accident_reports.clone();
                                for (track_id, accum_conf, bbox) in &accumulated_alerts {
                                    // Avoid duplicates: only add if no report already covers this bbox
                                    let already_reported = all_reports.iter().any(|r| {
                                        bbox_overlap(&r.location_bbox, bbox) > 0.5
                                    });
                                    if !already_reported {
                                        all_reports.push(AccidentReport {
                                            detected_at: timestamp,
                                            involved_ids: vec![],
                                            iou: 0.0,
                                            pred_iou: 0.0,
                                            speed_1: 0.0,
                                            speed_2: 0.0,
                                            accel_1: 0.0,
                                            accel_2: 0.0,
                                            location_bbox: *bbox,
                                            reasons: vec![
                                                format!(
                                                    "Tracker accumulated confidence {:.3} (track {})",
                                                    accum_conf, track_id
                                                ),
                                            ],
                                            confidence: if *accum_conf >= 3.0 {
                                                "very_high".to_string()
                                            } else if *accum_conf >= 2.0 {
                                                "high".to_string()
                                            } else {
                                                "medium".to_string()
                                            },
                                            track_id: Some(*track_id),
                                            accumulated_conf: Some(*accum_conf),
                                        });
                                    }
                                }

                                // --- Phase 7: Compute tracker-based traffic metrics ---
                                let traffic_metrics = {
                                    let pss = ss.lock().unwrap();
                                    crate::inference::compute_roi_traffic_metrics(
                                        &pss.tracker,
                                        &stream_rois,
                                        640,
                                        640,
                                        timestamp,
                                        None, // TODO: store previous occupancy for queue_growth_rate
                                        10.0, // 10s window
                                    )
                                };

                                // Also compute legacy vehicle counts for backwards compat
                                let vehicle_counts = crate::inference::compute_roi_vehicle_counts(
                                    &detections,
                                    &stream_rois,
                                    640,
                                    640,
                                );
                                let congestion_levels =
                                    crate::inference::compute_congestion_levels(&vehicle_counts);
                                let total_vehicles: u32 = vehicle_counts.values().copied().sum();

                                let rgb_img = image::RgbImage::from_raw(640, 640, frame_for_metrics)
                                    .unwrap_or_else(|| image::RgbImage::new(640, 640));
                                let traffic_lights = crate::inference::detect_traffic_lights_from_detections(
                                    &rgb_img,
                                    &detections,
                                    &stream_rois,
                                    640,
                                    640,
                                );

                                // --- Phase 8: Handle stalled vehicle events ---
                                handle_track_events_stalled(
                                    &track_events,
                                    &sid,
                                    timestamp,
                                    &client_inner,
                                    &state_inner,
                                )
                                .await;

                                // --- ACCIDENT REPORTING ---
                                if !all_reports.is_empty() {
                                    tracing::warn!(
                                        "[{}] ⚠️ ACCIDENT DETECTED! {} reports",
                                        sid,
                                        all_reports.len()
                                    );

                                    // Encode frame as JPEG for the report
                                    let frame_b64 = {
                                        let mut buf = std::io::Cursor::new(Vec::new());
                                        if let Err(_) = rgb_img.write_to(
                                            &mut buf,
                                            image::ImageOutputFormat::Jpeg(85),
                                        ) {
                                            None
                                        } else {
                                            Some(base64::Engine::encode(
                                                &base64::engine::general_purpose::STANDARD,
                                                buf.into_inner(),
                                            ))
                                        }
                                    };

                                    // Push confirmed accident reports to scheduler
                                    let accidents_url = format!(
                                        "{}/api/v1/accidents",
                                        state_inner.scheduler_url
                                    );
                                    for report in &all_reports {
                                        let accident_payload = serde_json::json!({
                                            "type": "accident_alert",
                                            "camera_name": sid.clone(),
                                            "timestamp": timestamp,
                                            "vehicle_counts": vehicle_counts,
                                            "congestion_levels": congestion_levels,
                                            "total_vehicles": total_vehicles,
                                            "tracked_vehicles": track_count,
                                            "details": report,
                                            "frame": frame_b64,
                                        });

                                        // Persist the frame + metadata locally for inspection.
                                        let debug_context = serde_json::json!({
                                            "vehicle_counts": vehicle_counts,
                                            "congestion_levels": congestion_levels,
                                            "total_vehicles": total_vehicles,
                                            "tracked_vehicles": track_count,
                                            "timestamp": timestamp,
                                            "worker_id": state_inner.worker_id.as_str(),
                                            "dark_ratio": detection_summary.dark_ratio,
                                            "used_night_enhancement": detection_summary.used_night_enhancement,
                                            "max_accident_confidence": detection_summary.max_accident_confidence,
                                        });
                                        crate::accident_debug::save_accident_snapshot(
                                            &sid,
                                            &rgb_img,
                                            report,
                                            &debug_context,
                                        );

                                        let _ = client_inner
                                            .post(&accidents_url)
                                            .json(&accident_payload)
                                            .send()
                                            .await;
                                    }
                                }

                                // --- HIGH CONGESTION REPORTING ---
                                let has_high = congestion_levels.values().any(|level| level == "High" || level == "Severe");
                                let can_alert_congestion = {
                                    let mut pss = ss.lock().unwrap();
                                    if has_high && (timestamp - pss.congestion_cooldown_until) > 0.0 {
                                        pss.congestion_cooldown_until = timestamp + 60.0; // 1 minute cooldown
                                        true
                                    } else {
                                        false
                                    }
                                };

                                if can_alert_congestion {
                                    tracing::warn!("[{}] 🚦 HIGH CONGESTION DETECTED! Dispatching alert.", sid);
                                    
                                    let frame_b64 = {
                                        let mut buf = std::io::Cursor::new(Vec::new());
                                        if let Ok(_) = rgb_img.write_to(&mut buf, image::ImageOutputFormat::Jpeg(85)) {
                                            Some(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, buf.into_inner()))
                                        } else {
                                            None
                                        }
                                    };

                                    let congestion_payload = serde_json::json!({
                                        "type": "congestion_alert",
                                        "camera_name": sid.clone(),
                                        "timestamp": timestamp,
                                        "congestion_levels": congestion_levels,
                                        "vehicle_counts": vehicle_counts,
                                        "total_vehicles": total_vehicles,
                                        "tracked_vehicles": track_count,
                                        "traffic_metrics": traffic_metrics,
                                        "frame": frame_b64,
                                    });

                                    let congestion_url = format!(
                                        "{}/api/v1/congestion",
                                        state_inner.scheduler_url
                                    );
                                    let _ = client_inner
                                        .post(&congestion_url)
                                        .json(&congestion_payload)
                                        .send()
                                        .await;
                                }

                                let payload = serde_json::json!({
                                    "camera_name": sid.clone(),
                                    "vehicle_counts": vehicle_counts,
                                    "congestion_levels": congestion_levels,
                                    "total_vehicles": total_vehicles,
                                    "tracked_vehicles": track_count,
                                    "traffic_lights": traffic_lights,
                                    "traffic_metrics": traffic_metrics
                                });

                                let metrics_url = format!(
                                    "{}/api/v1/metrics",
                                    state_inner.scheduler_url
                                );
                                match client_inner
                                    .post(&metrics_url)
                                    .json(&payload)
                                    .send()
                                    .await
                                {
                                    Ok(res) if res.status().is_success() => {
                                        info!(
                                            "[{}] Processed frame {}. Detections: {}. Tracks: {}. Metrics stored.",
                                            sid,
                                            fidx,
                                            detections.len(),
                                            track_count,
                                        );
                                    }
                                    Ok(res) => {
                                        error!(
                                            "[{}] Metrics push failed with status {}",
                                            sid,
                                            res.status()
                                        );
                                    }
                                    Err(e) => {
                                        error!("[{}] Failed to push metrics to scheduler: {}", sid, e);
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            consecutive_failures += 1;
                            error!("FFmpeg stream broke for {}: {}", stream_id, e);

                            let status_payload = serde_json::json!({
                                "worker_id": state.worker_id.as_str(),
                                "stream_id": stream_id,
                                "status": if consecutive_failures >= 20 { "failed" } else { "degraded" },
                                "reason": e.to_string(),
                            });

                            let status_url = format!(
                                "{}/api/v1/nodes/stream-status",
                                state.scheduler_url
                            );
                            let _ = scheduler_client
                                .post(&status_url)
                                .json(&status_payload)
                                .send()
                                .await;

                            break;
                        }
                    }
                }

                let _ = child.kill().await;
                let _ = child.wait().await;

                if consecutive_failures >= 20 {
                    error!(
                        "Stream {} marked permanently failed after repeated reconnect errors",
                        stream_id
                    );
                    break;
                }

                if !got_any_frame {
                    sleep_with_backoff(&stream_id, consecutive_failures).await;
                } else {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
            }

            {
                let mut running = state.running_streams.write().unwrap();
                running.remove(&stream_id);
            }
            state.active_streams.fetch_sub(1, Ordering::SeqCst);
        });
    }
}

// ---------------------------------------------------------------------------
// Phase 8: Handle stalled vehicle events from the tracker
// ---------------------------------------------------------------------------

async fn handle_track_events_stalled(
    events: &[TrackEvent],
    stream_id: &str,
    timestamp: f64,
    client: &reqwest::Client,
    state: &WorkerState,
) {
    for event in events {
        if let TrackEvent::Stalled {
            track_id,
            dwell_sec,
            bbox,
        } = event
        {
            let report =
                StalledVehicleReport::from_stall(*track_id, *bbox, *dwell_sec, timestamp);
            tracing::warn!(
                "[{}] 🚗 STALLED VEHICLE: track {} stationary for {:.1}s",
                stream_id,
                track_id,
                dwell_sec
            );

            let stalled_url = format!("{}/api/v1/stalled", state.scheduler_url);
            let payload = serde_json::json!({
                "type": "stalled_vehicle",
                "camera_name": stream_id,
                "timestamp": timestamp,
                "report": report,
            });

            let _ = client.post(&stalled_url).json(&payload).send().await;
        }
    }
}

// ---------------------------------------------------------------------------
// Detection helpers (unchanged from original)
// ---------------------------------------------------------------------------

fn detect_with_optional_enhancement(
    state: &WorkerState,
    rgb_640: &[u8],
    timestamp: f64,
) -> DetectionSummary {
    let dark_ratio = crate::night_enhance::dark_ratio(rgb_640);
    let is_night = dark_ratio >= crate::night_enhance::NIGHT_RATIO_THRESH;
    let mut merged = run_detector(state, rgb_640);
    let mut used_night_enhancement = false;

    if is_night {
        let enhanced = state.night_enhancer.enhance(rgb_640);
        let enhanced_detections = run_detector(state, &enhanced);
        if !enhanced_detections.is_empty() {
            used_night_enhancement = true;
            merged.extend(enhanced_detections);
            merged = crate::inference::apply_nms(merged, alpen_detect::ALPEN_DETECT_NMS_IOU);
        }
    }

    alpen_detect::summarize_detections(merged, timestamp, used_night_enhancement, dark_ratio)
}

fn run_detector(state: &WorkerState, rgb_640: &[u8]) -> Vec<[f32; 6]> {
    let model_arc = loop {
        if let Some(s) = state.pool.pop() {
            break s;
        }
        std::thread::sleep(Duration::from_millis(1));
    };

    let result = {
        let mut model = model_arc.lock().unwrap();
        alpen_detect::run_session_augmented(&mut model, rgb_640).unwrap_or_default()
    };
    let _ = state.pool.push(model_arc);
    result
}

fn normalize_stream_url(url: &str) -> String {
    if url.ends_with(".stream") {
        return format!("{}/playlist.m3u8", url);
    }
    if url.ends_with(".stream/") {
        return format!("{}playlist.m3u8", url);
    }
    url.to_string()
}

async fn sleep_with_backoff(stream_id: &str, failures: u32) {
    let exp = failures.min(6);
    let base_ms = 1_000u64.saturating_mul(1u64 << exp);
    let cap_ms = 30_000u64;
    let bounded = base_ms.min(cap_ms);
    let hash = stream_id.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
    let jitter = (hash + failures as u64 * 137) % 700;
    tokio::time::sleep(Duration::from_millis(bounded + jitter)).await;
}

/// Quick bbox overlap ratio (used to deduplicate tracker-based and frame-based
/// accident reports).
fn bbox_overlap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}
