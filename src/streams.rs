use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Semaphore;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tracing::{info, error};
use crate::state::WorkerState;

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
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "640x640",
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
                            let rgb_data = buffer.clone();
                            let state_inner = state.clone();
                            let sid = stream_id.clone();
                            let client_inner = metrics_client.clone();

                            tokio::spawn(async move {
                                let frame_for_metrics = rgb_data.clone();
                                let input = ndarray::Array::from_shape_vec((640, 640, 3), rgb_data)
                                    .expect("Failed to reshape raw ffmpeg data")
                                    .mapv(|v| v as f32 / 255.0)
                                    .permuted_axes([2, 0, 1])
                                    .insert_axis(ndarray::Axis(0))
                                    .into_dyn();

                                let model_arc = loop {
                                    if let Some(s) = state_inner.pool.pop() {
                                        break s;
                                    }
                                    tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                                };

                                let result = {
                                    let mut model = model_arc.lock().unwrap();
                                    let mut io_binding =
                                        model.create_binding().expect("Failed to create binding");

                                    let input_tensor = ort::value::Value::from_array(input).unwrap();

                                    io_binding
                                        .bind_input("images", &input_tensor)
                                        .expect("Failed to bind input");
                                    io_binding
                                        .bind_output_to_device("output0", model.allocator().memory_info())
                                        .expect("Failed to bind output");

                                    let outputs = model
                                        .run_binding(&io_binding)
                                        .expect("FFmpeg Inference failed");
                                    outputs["output0"].try_extract_tensor::<f32>().unwrap().1.to_vec()
                                };

                                state_inner.pool.push(model_arc).ok();

                                let decoded = crate::inference::decode_output(&result);
                                let detections = crate::inference::apply_nms(decoded, 0.45);
                                state_inner.record_inference(&sid);

                                let stream_rois = {
                                    let rois = state_inner.stream_rois.read().unwrap();
                                    rois.get(&sid).cloned().unwrap_or_default()
                                };

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

                                let payload = serde_json::json!({
                                    "camera_name": sid.clone(),
                                    "vehicle_counts": vehicle_counts,
                                    "congestion_levels": congestion_levels,
                                    "total_vehicles": total_vehicles,
                                    "traffic_lights": traffic_lights
                                });

                                match client_inner
                                    .post("http://127.0.0.1:3000/api/v1/metrics")
                                    .json(&payload)
                                    .send()
                                    .await
                                {
                                    Ok(res) if res.status().is_success() => {
                                        info!(
                                            "[{}] Processed frame. Detections: {}. Metrics stored.",
                                            sid,
                                            detections.len()
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
                                "worker_id": "gpu-worker-1",
                                "stream_id": stream_id,
                                "status": if consecutive_failures >= 20 { "failed" } else { "degraded" },
                                "reason": e.to_string(),
                            });

                            let _ = scheduler_client
                                .post("http://127.0.0.1:3000/api/v1/nodes/stream-status")
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
