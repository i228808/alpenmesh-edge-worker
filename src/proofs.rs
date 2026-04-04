use crate::state::{ProofBatch, WorkerState};
use tracing::{error, info};

const FLUSH_INTERVAL_SECS: u64 = 5;
const SCHEDULER_PROOFS_ENDPOINT: &str = "http://127.0.0.1:3000/api/v1/proofs";
const ECONOMY_REGISTER_ENDPOINT: &str = "http://127.0.0.1:3010/api/v1/economy/workers/register";

pub fn start_proof_flush_task(state: WorkerState) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime for proof flush task");

        rt.block_on(async move {
            let client = reqwest::Client::new();
            if let Err(err) = register_worker_identity(&client, &state.worker_id, &state.worker_secret).await {
                error!("Worker identity registration failed: {}", err);
            }

            loop {
                tokio::time::sleep(std::time::Duration::from_secs(FLUSH_INTERVAL_SECS)).await;

                let Some(batch) = state.drain_proof_batch() else {
                    continue;
                };

                if let Err(err) = post_proof_batch(&client, &state.worker_id, batch).await {
                    error!("Proof batch push failed: {}", err);
                }
            }
        });
    });
}

async fn register_worker_identity(
    client: &reqwest::Client,
    worker_id: &str,
    worker_secret: &str,
) -> Result<(), reqwest::Error> {
    let payload = serde_json::json!({
        "worker_id": worker_id,
        "worker_secret": worker_secret,
    });

    let response = client
        .post(ECONOMY_REGISTER_ENDPOINT)
        .json(&payload)
        .send()
        .await?;

    if response.status().is_success() {
        info!("Worker identity registered in alpenmesh economy service");
    } else {
        error!("Worker registration rejected with status {}", response.status());
    }

    Ok(())
}

async fn post_proof_batch(
    client: &reqwest::Client,
    worker_id: &str,
    batch: ProofBatch,
) -> Result<(), reqwest::Error> {
    let payload = serde_json::json!({
        "worker_id": worker_id,
        "window_start_ms": batch.window_start_ms,
        "window_end_ms": batch.window_end_ms,
        "total_inferences": batch.total_inferences,
        "camera_inferences": batch.camera_inferences,
    });

    let response = client.post(SCHEDULER_PROOFS_ENDPOINT).json(&payload).send().await?;
    if response.status().is_success() {
        info!(
            "Proof batch sent: total_inferences={} window_start_ms={} window_end_ms={}",
            batch.total_inferences,
            batch.window_start_ms,
            batch.window_end_ms
        );
    } else {
        error!("Proof batch rejected with status {}", response.status());
    }

    Ok(())
}
