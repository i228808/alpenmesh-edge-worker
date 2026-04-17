use std::thread;
use std::time::Duration;
use std::sync::atomic::Ordering;
use tracing::{info, error};
use nvml_wrapper::Nvml;
use crate::state::WorkerState;

pub fn start_heartbeat_task(state: WorkerState) {
    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime for heartbeat");

        rt.block_on(async {
            let client = reqwest::Client::new();
            info!("Started background heartbeat task with telemetry");

            // Initialize NVML for real-time VRAM tracking
            let nvml = Nvml::init().expect("Failed to initialize NVML for heartbeat");
            let device = nvml.device_by_index(0).expect("Failed to get GPU for heartbeat");

            loop {
                // Query real-time specs
                let mem = device.memory_info().unwrap_or_else(|_| {
                    error!("Failed to query GPU memory for heartbeat");
                    nvml_wrapper::struct_wrappers::device::MemoryInfo { total: 0, free: 0, used: 0 }
                });

                let active_streams = state.active_streams.load(Ordering::Relaxed);

                let payload = serde_json::json!({
                    "id": state.worker_id.as_str(),
                    "ip_address": state.external_addr.as_str(),
                    "total_vram": state.total_vram_mb,
                    "free_vram": mem.free / 1024 / 1024,
                    "max_lanes": state.pool_size,
                    "active_streams": active_streams
                });

                let heartbeat_url = format!("{}/api/v1/nodes/heartbeat", state.scheduler_url);
                match client.post(&heartbeat_url)
                    .json(&payload)
                    .send()
                    .await 
                {
                    Ok(res) if res.status().is_success() => {
                        info!("Heartbeat sent: {}mb free, {}/{} lanes active", 
                            mem.free / 1024 / 1024, active_streams, state.pool_size);
                    }
                    Ok(res) => {
                        error!("Heartbeat failed with status: {}", res.status());
                    }
                    Err(e) => {
                        error!("Heartbeat failed to send: {}", e);
                    }
                }

                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    });
}
