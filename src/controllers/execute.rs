use oxide_framework_core::{controller, ApiResponse, AppState, Data};
use axum::extract::Multipart;
use std::time::Instant;
use tracing::info;

use crate::state::WorkerState;
use crate::alpen_detect::{self, DetectionSummary};
use crate::inference::RoiMap;

pub struct ExecuteController;

#[controller("/")]
impl ExecuteController {
    fn new(_state: &AppState) -> Self {
        Self
    }

    #[post("/execute")]
    async fn execute(
        &self,
        Data(state): Data<WorkerState>,
        headers: axum::http::HeaderMap,
        mut multipart: Multipart,
    ) -> ApiResponse<serde_json::Value> {
        let camera_id = headers
            .get("X-Camera-ID")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown");

        let mut frame_bytes: Vec<u8> = Vec::new();
        let mut camera_name = camera_id.to_string();
        let mut parsed_rois: RoiMap = RoiMap::new();

        loop {
            match multipart.next_field().await {
                Ok(Some(field)) => {
                    let name = field.name().unwrap_or("").to_string();
                    if name == "frame" {
                        match field.bytes().await {
                            Ok(bytes) => frame_bytes = bytes.to_vec(),
                            Err(_) => {
                                return ApiResponse::error(oxide_framework_core::StatusCode::BAD_REQUEST, "Invalid frame field")
                            }
                        }
                    } else if name == "camera_name" {
                        if let Ok(text) = field.text().await {
                            if !text.trim().is_empty() {
                                camera_name = text;
                            }
                        }
                    } else if name == "rois" {
                        if let Ok(text) = field.text().await {
                            if let Ok(map) = serde_json::from_str::<RoiMap>(&text) {
                                parsed_rois = map;
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    return ApiResponse::error(oxide_framework_core::StatusCode::BAD_REQUEST, "Invalid multipart payload")
                }
            }
        }

        if frame_bytes.is_empty() {
            return ApiResponse::error(oxide_framework_core::StatusCode::BAD_REQUEST, "Empty body");
        }

        if parsed_rois.is_empty() {
            let rois = state.stream_rois.read().unwrap();
            if let Some(map) = rois.get(&camera_name) {
                parsed_rois = map.clone();
            } else if let Some(map) = rois.get(camera_id) {
                parsed_rois = map.clone();
            }
        }

        let start = Instant::now();

        // 1. High-Speed Native Pre-processing (zune-jpeg + fast-image-resize)
        let mut decoder = zune_jpeg::JpegDecoder::new(&frame_bytes);
        let pixels = decoder.decode().expect("Failed to decode JPEG");
        let info = decoder.info().unwrap();
        let original_w = info.width as u32;
        let original_h = info.height as u32;

        let original_rgb = match image::load_from_memory(&frame_bytes) {
            Ok(img) => img.to_rgb8(),
            Err(_) => image::RgbImage::new(original_w.max(1), original_h.max(1)),
        };
        
        use fast_image_resize as fr;
        use std::num::NonZeroU32;
        
        let src_width = NonZeroU32::new(info.width as u32).unwrap();
        let src_height = NonZeroU32::new(info.height as u32).unwrap();
        
        // In v3.0, Image is at the top level of the crate
        let src_image = fr::Image::from_vec_u8(
            src_width, 
            src_height, 
            pixels, 
            fr::PixelType::U8x3
        ).expect("Failed to create source image");

        let dst_width = NonZeroU32::new(640).unwrap();
        let dst_height = NonZeroU32::new(640).unwrap();
        let mut dst_image = fr::Image::new(dst_width, dst_height, src_image.pixel_type());

        // Resizer::new() requires a ResizeAlg in v3.0
        let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        resizer.resize(&src_image.view(), &mut dst_image.view_mut()).expect("Failed to resize image");

        let rgb_data = dst_image.into_vec();
        let duration = start.elapsed();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let detection_summary = detect_with_optional_enhancement(&state, &rgb_data, timestamp);
        let final_boxes = detection_summary.detections.clone();
        state.record_inference(&camera_name);

        // 5. Visual Debug Capture (Background task, non-blocking)
        let debug_count = state.debug_capture_count.load(std::sync::atomic::Ordering::Relaxed);
        if debug_count < 20 && !final_boxes.is_empty() {
            state.debug_capture_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let boxes = final_boxes.clone();
            let raw_bytes = frame_bytes.clone();
            let count = debug_count + 1;
            
            tokio::spawn(async move {
                if let Ok(img) = image::load_from_memory(&raw_bytes) {
                    let mut img = img.to_rgb8();
                    let (w, h) = img.dimensions();
                    
                    for b in boxes {
                        // Rescale coordinates from 640x640 to original image size
                        let x1 = (b[0] * w as f32 / 640.0) as u32;
                        let y1 = (b[1] * h as f32 / 640.0) as u32;
                        let x2 = (b[2] * w as f32 / 640.0) as u32;
                        let y2 = (b[3] * h as f32 / 640.0) as u32;
                        
                        // Draw a simple red box (manually for speed and to avoid extra deps)
                        let red = image::Rgb([255, 0, 0]);
                        for x in x1.max(0)..x2.min(w - 1) {
                            if y1 < h { img.put_pixel(x, y1, red); }
                            if y2 < h { img.put_pixel(x, y2, red); }
                        }
                        for y in y1.max(0)..y2.min(h - 1) {
                            if x1 < w { img.put_pixel(x1, y, red); }
                            if x2 < w { img.put_pixel(x2, y, red); }
                        }
                    }
                    let _ = img.save(format!("debug_outputs/detection_{}.jpg", count));
                }
            });
        }

        let vehicle_counts = crate::inference::compute_roi_vehicle_counts(
            &final_boxes,
            &parsed_rois,
            original_w.max(1),
            original_h.max(1),
        );
        let congestion_levels = crate::inference::compute_congestion_levels(&vehicle_counts);
        let total_vehicles: u32 = vehicle_counts.values().copied().sum();
        let traffic_lights = crate::inference::detect_traffic_lights_from_detections(
            &original_rgb,
            &final_boxes,
            &parsed_rois,
            original_w.max(1),
            original_h.max(1),
        );

        // Persist each confirmed accident frame + metadata under debug/accidents/.
        if !detection_summary.accident_reports.is_empty() {
            let debug_context = serde_json::json!({
                "vehicle_counts": vehicle_counts,
                "congestion_levels": congestion_levels,
                "total_vehicles": total_vehicles,
                "timestamp": timestamp,
                "worker_id": state.worker_id.as_str(),
                "source": "execute_controller",
                "dark_ratio": detection_summary.dark_ratio,
                "used_night_enhancement": detection_summary.used_night_enhancement,
                "max_accident_confidence": detection_summary.max_accident_confidence,
            });
            for report in &detection_summary.accident_reports {
                crate::accident_debug::save_accident_snapshot(
                    &camera_name,
                    &original_rgb,
                    report,
                    &debug_context,
                );
            }
        }

        let has_high = congestion_levels.values().any(|level| level == "High" || level == "Severe");
        let mut congestion_payload = None;
        let can_alert_congestion = {
            let mut cooldowns = state.congestion_cooldowns.write().unwrap();
            let last_sent = cooldowns.get(&camera_name).copied().unwrap_or(0.0);
            if has_high && (timestamp - last_sent) > 60.0 { // 1 minute
                cooldowns.insert(camera_name.clone(), timestamp);
                true
            } else {
                false
            }
        };

        if can_alert_congestion {
            tracing::warn!("[{}] 🚦 HIGH CONGESTION DETECTED! Including in execution response.", camera_name);
            let mut buf = std::io::Cursor::new(Vec::new());
            if let Ok(_) = original_rgb.write_to(&mut buf, image::ImageOutputFormat::Jpeg(85)) {
                let frame_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, buf.into_inner());
                congestion_payload = Some(serde_json::json!({
                    "type": "congestion_alert",
                    "camera_name": camera_name.clone(),
                    "timestamp": timestamp,
                    "congestion_levels": congestion_levels,
                    "vehicle_counts": vehicle_counts,
                    "total_vehicles": total_vehicles,
                    "frame": frame_b64,
                }));
            }
        }

        info!("[{}] Peak Inference completed in {}ms. Found {} detections.", camera_name, duration.as_millis(), final_boxes.len());
        
        ApiResponse::ok(serde_json::json!({
            "camera_name": camera_name,
            "detections": final_boxes,
            "latency": duration.as_millis(),
            "vehicle_counts": vehicle_counts,
            "congestion_levels": congestion_levels,
            "total_vehicles": total_vehicles,
            "traffic_lights": traffic_lights,
            "accidents": detection_summary.accident_reports,
            "dark_ratio": detection_summary.dark_ratio,
            "used_night_enhancement": detection_summary.used_night_enhancement,
            "max_accident_confidence": detection_summary.max_accident_confidence,
            "congestion_alert": congestion_payload
        }))    
    }
}

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
            merged.extend(enhanced_detections);
            merged = crate::inference::apply_nms(merged, alpen_detect::ALPEN_DETECT_NMS_IOU);
            used_night_enhancement = true;
        }
    }

    alpen_detect::summarize_detections(merged, timestamp, used_night_enhancement, dark_ratio)
}

fn run_detector(state: &WorkerState, rgb_640: &[u8]) -> Vec<[f32; 6]> {
    let model_arc = loop {
        if let Some(s) = state.pool.pop() {
            break s;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    };

    let result = {
        let mut model = model_arc.lock().expect("Failed to lock model session");
        alpen_detect::run_session_augmented(&mut model, rgb_640).unwrap_or_default()
    };

    state.pool.push(model_arc).expect("Failed to return session to pool");
    result
}
