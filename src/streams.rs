use tokio::process::{Command, Child};
use tokio::io::AsyncReadExt;
use std::sync::atomic::Ordering;
use tracing::{info, error};
use crate::state::WorkerState;

pub struct StreamManager {
    pub state: WorkerState,
}

impl StreamManager {
    pub fn new(state: WorkerState) -> Self {
        Self { state }
    }

    pub async fn spawn_stream(&self, stream_id: String, url: String) {
        let state = self.state.clone();
        
        tokio::spawn(async move {
            info!("Spawning Native Ingestion for stream {}: {}", stream_id, url);
            state.active_streams.fetch_add(1, Ordering::SeqCst);

            let mut child: Child = Command::new("ffmpeg")
                .args([
                    "-rtsp_transport", "tcp",
                    "-hwaccel", "nvdec",
                    "-i", &url,
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-s", "640x640",
                    "-r", "5",
                    "-an",
                    "-"
                ])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("Failed to spawn ffmpeg. Is it installed?");

            let mut stdout = child.stdout.take().expect("Failed to open ffmpeg stdout");
            let frame_size = 640 * 640 * 3;
            let mut buffer = vec![0u8; frame_size];

            loop {
                match stdout.read_exact(&mut buffer).await {
                    Ok(_) => {
                        let rgb_data = buffer.clone();
                        let state_inner = state.clone();
                        let sid = stream_id.clone();

                        tokio::spawn(async move {
                            let input = ndarray::Array::from_shape_vec((640, 640, 3), rgb_data)
                                .expect("Failed to reshape raw ffmpeg data")
                                .mapv(|v| v as f32 / 255.0)
                                .permuted_axes([2, 0, 1])
                                .insert_axis(ndarray::Axis(0))
                                .into_dyn();

                            let model_arc = loop {
                                if let Some(s) = state_inner.pool.pop() { break s; }
                                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                            };

                            let result = {
                                let mut model = model_arc.lock().unwrap();
                                let mut io_binding = model.create_binding().expect("Failed to create binding");
                                
                                // Explicitly use ort::value::Value for rc.12 compatibility
                                let input_tensor = ort::value::Value::from_array(input).unwrap();
                                
                                io_binding.bind_input("images", &input_tensor).expect("Failed to bind input");
                                io_binding.bind_output_to_device("output0", model.allocator().memory_info()).expect("Failed to bind output");
                                
                                let outputs = model.run_binding(&io_binding).expect("FFmpeg Inference failed");
                                outputs["output0"].try_extract_tensor::<f32>().unwrap().1.to_vec()
                            };

                            state_inner.pool.push(model_arc).ok();

                            let detections = crate::inference::decode_output(&result);
                            if !detections.is_empty() {
                                info!("[{}] Detected {} objects natively.", sid, detections.len());
                            }
                        });
                    }
                    Err(e) => {
                        error!("FFmpeg stream broke for {}: {}", stream_id, e);
                        break;
                    }
                }
            }

            state.active_streams.fetch_sub(1, Ordering::SeqCst);
        });
    }
}
