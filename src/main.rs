use oxide_framework_core::App;
use alpenmesh_worker::{controllers, heartbeat, identity, proofs, state::WorkerState};

fn main() {
    // 0. Load .env (if present). WORKER_ENV=<path> picks a specific file, letting
    // multiple workers on one host each load their own environment.
    if let Ok(custom) = std::env::var("WORKER_ENV") {
        let _ = dotenvy::from_filename(&custom);
    } else {
        let _ = dotenvy::dotenv();
    }

    // 1. Critical PATH Injection for NVIDIA/TensorRT DLLs
    #[cfg(target_os = "windows")]
    {
        let trt_path = "C:\\lib\\TensorRT\\bin";
        let cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.2\\bin";

        if let Ok(current_path) = std::env::var("PATH") {
            let new_path = format!("{};{};{}", trt_path, cuda_path, current_path);
            unsafe {
                std::env::set_var("PATH", new_path);
            }
        }
    }

    // 1. Initialize GPU inference state (with hardware check)
    let state = WorkerState::new();
    let identity_path = identity::identity_file_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "<unresolved>".to_string());
    println!("🔑 Worker ID: {}", state.worker_id);
    println!(
        "🧾 Worker Secret: ********{}",
        state
            .worker_secret
            .chars()
            .rev()
            .take(6)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );
    println!("💾 Worker Identity File: {}", identity_path);

    // 2. Initialize background tokio task for heartbeat with telemetry
    heartbeat::start_heartbeat_task(state.clone());
    proofs::start_proof_flush_task(state.clone());

    println!("--------------------------------------------------");
    println!("🚀 ALPENMESH WORKER STARTING: SPECS-AWARE NATIVE INGESTION");
    println!("--------------------------------------------------");

    // Allow running multiple workers on one host by pointing each at its own
    // config file (different listener port). Defaults to "app.yaml".
    let config_path = std::env::var("WORKER_CONFIG").unwrap_or_else(|_| "app.yaml".to_string());
    println!("📄 Worker Config: {}", config_path);
    println!("🌐 Worker External Addr: {}", state.external_addr);
    println!("📡 Scheduler URL: {}", state.scheduler_url);

    App::new()
        .config(&config_path)
        .state(state)
        .controller::<controllers::ExecuteController>()
        .controller::<controllers::StreamController>()
        .run();
}
