use oxide_framework_core::{controller, ApiResponse, AppState, Data};
use axum::body::Bytes;
use std::time::Instant;
use tracing::info;

use crate::state::WorkerState;

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
        body: Bytes,
    ) -> ApiResponse<serde_json::Value> {
        let camera_id = headers
            .get("X-Camera-ID")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown");

        if body.is_empty() {
            return ApiResponse::error(oxide_framework_core::StatusCode::BAD_REQUEST, "Empty body");
        }

        let frame_bytes = body.to_vec();
        let start = Instant::now();

        // 1. High-Speed Native Pre-processing (zune-jpeg + fast-image-resize)
        let mut decoder = zune_jpeg::JpegDecoder::new(&frame_bytes);
        let pixels = decoder.decode().expect("Failed to decode JPEG");
        let info = decoder.info().unwrap();
        
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

        // 2. Vectorized Normalization and Reshape (HWC -> CHW)
        let rgb_data = dst_image.into_vec();
        let input = ndarray::Array::from_shape_vec((640, 640, 3), rgb_data)
            .expect("Failed to reshape image data")
            .mapv(|v| v as f32 / 255.0)
            .permuted_axes([2, 0, 1])
            .insert_axis(ndarray::Axis(0))
            .to_owned();

        // 3. Inference & I/O Binding
        let input_tensor = match ort::value::Value::from_array(input) {
            Ok(v) => v,
            Err(e) => return ApiResponse::internal_error(format!("Failed to create input tensor: {}", e)),
        };

        // 3b. High-Performance Session Acquisition from Pool
        let model_arc = loop {
            if let Some(s) = state.pool.pop() {
                break s;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        };

        let result = {
            let mut model = model_arc.lock().expect("Failed to lock model session");
            let mut io_binding = model.create_binding().expect("Failed to create binding");
            
            io_binding.bind_input("images", &input_tensor).expect("Failed to bind input");
            io_binding.bind_output_to_device("output0", model.allocator().memory_info()).expect("Failed to bind output");

            let outputs = model.run_binding(&io_binding).expect("Model execution with I/O binding failed");

            match outputs["output0"].try_extract_tensor::<f32>() {
                Ok(tensor) => tensor.1.to_vec(), 
                Err(e) => return ApiResponse::internal_error(format!("Failed to extract output: {}", e)),
            }
        };

        state.pool.push(model_arc).expect("Failed to return session to pool");

        let duration = start.elapsed();
        
        // 4. Post-processing (Filtering & NMS)
        let decoded = crate::inference::decode_output(&result);
        let final_boxes = crate::inference::apply_nms(decoded, 0.45);

        // 5. Visual Debug Capture (Background task, non-blocking)
        let debug_count = state.debug_capture_count.load(std::sync::atomic::Ordering::Relaxed);
        if debug_count < 20 && !final_boxes.is_empty() {
            state.debug_capture_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let boxes = final_boxes.clone();
            let raw_bytes = frame_bytes.clone();
            let count = debug_count + 1;
            
            tokio::spawn(async move {
                if let Ok(mut img) = image::load_from_memory(&raw_bytes) {
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

        info!("[{}] Peak Inference completed in {}ms. Found {} detections.", camera_id, duration.as_millis(), final_boxes.len());
        
        ApiResponse::ok(serde_json::json!({
            "detections": final_boxes,
            "latency": duration.as_millis()
        }))
    }
}
