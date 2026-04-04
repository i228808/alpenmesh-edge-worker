use oxide_framework_core::App;
mod controllers;
mod heartbeat;
mod state;
mod inference;
mod streams;
mod proofs;

use crate::state::WorkerState;

fn main() {
    // 1. Critical PATH Injection for NVIDIA/TensorRT DLLs
    let trt_path = "C:\\lib\\TensorRT\\bin";
    let cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin";
    
    if let Ok(current_path) = std::env::var("PATH") {
        let new_path = format!("{};{};{}", trt_path, cuda_path, current_path);
        unsafe {
            std::env::set_var("PATH", new_path);
        }
    }

    // 1. Initialize GPU inference state (with hardware check)
    let state = WorkerState::new();
    println!("🔑 Worker ID: {}", state.worker_id);

    // 2. Initialize background tokio task for heartbeat with telemetry
    heartbeat::start_heartbeat_task(state.clone());
    proofs::start_proof_flush_task(state.clone());

    println!("--------------------------------------------------");
    println!("🚀 OXIDE WORKER STARTING: SPECS-AWARE NATIVE INGESTION");
    println!("--------------------------------------------------");

    App::new()
        .config("app.yaml")
        .state(state)
        .controller::<controllers::ExecuteController>()
        .controller::<controllers::StreamController>()
        .run();
}
