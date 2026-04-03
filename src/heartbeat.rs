use std::thread;
use std::time::Duration;
use tracing::{info, error};

pub fn start_heartbeat_task() {
    thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime for heartbeat");

        rt.block_on(async {
            let client = reqwest::Client::new();
            info!("Started background heartbeat task");

            loop {
                let payload = serde_json::json!({
                    "id": "gpu-worker-1",
                    "ip_address": "127.0.0.1:8081",
                    "vram_available": 16384_u64
                });

                match client.post("http://127.0.0.1:3000/api/v1/nodes/heartbeat")
                    .json(&payload)
                    .send()
                    .await 
                {
                    Ok(res) if res.status().is_success() => {
                        info!("Heartbeat sent successfully");
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
