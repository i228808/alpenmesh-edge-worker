use oxide_framework_core::App;
mod controllers;
mod heartbeat;
mod state;
mod inference;

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

    // Initialize background tokio task for heartbeat
    heartbeat::start_heartbeat_task();

    // Initialize GPU inference state (with hardware check)
    let state = WorkerState::new();

    println!("--------------------------------------------------");
    println!("🚀 OXIDE WORKER STARTING");
    println!("--------------------------------------------------");

    App::new()
        .config("app.yaml")
        .state(state)
        .controller::<controllers::ExecuteController>()
        .run();
}
