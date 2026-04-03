use crate::inference::RoiMap;
use crossbeam_queue::ArrayQueue;
use nvml_wrapper::Nvml;
use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::RwLock;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct WorkerState {
    /// High-performance queue of Session instances.
    /// Acts as a 'pool' of parallel inference lanes.
    pub pool: Arc<ArrayQueue<Arc<Mutex<Session>>>>,
    /// Counter for debug image captures
    pub debug_capture_count: Arc<AtomicUsize>,
    /// Total number of parallel sessions initialized
    pub pool_size: usize,
    /// Total GPU VRAM detected in MB
    pub total_vram_mb: u64,
    /// Number of active RTSP streams being processed
    pub active_streams: Arc<AtomicUsize>,
    /// ROI polygons keyed by stream/camera id
    pub stream_rois: Arc<RwLock<HashMap<String, RoiMap>>>,
}

impl WorkerState {
    pub fn new() -> Self {
        println!("--------------------------------------------------");
        println!("🚀 HARDWARE DIAGNOSTICS: INITIALIZING GPU POOL");
        println!("--------------------------------------------------");

        let nvml = Nvml::init().expect("Failed to initialize NVML");
        let device = nvml.device_by_index(0).expect("Failed to get GPU device");
        let mem = device.memory_info().expect("Failed to get memory info");
        let total_vram_mb = mem.total / 1024 / 1024;

        // 1. Detect VRAM using NVML
        let pool_size = match Self::calculate_pool_size(&device) {
            Ok(size) => {
                println!(
                    "[GPU Check] Available VRAM detected. Initializing {} parallel lanes.",
                    size
                );
                size
            }
            Err(e) => {
                println!(
                    "[GPU Warning] Hardware check failed: {}. Falling back to 1 lane.",
                    e
                );
                1
            }
        };

        // 2. Prepare Cache
        std::fs::create_dir_all("./assets/cache").expect("Failed to create assets/cache directory");

        // 3. Initialize Pool
        let pool = ArrayQueue::new(pool_size);

        println!(
            "[GPU Ready] Building {} Parallel Sessions (TensorRT/CUDA)...",
            pool_size
        );

        for i in 0..pool_size {
            let session = Self::create_session();
            pool.push(Arc::new(Mutex::new(session)))
                .expect("Failed to fill session pool");
            if (i + 1) % 5 == 0 || i + 1 == pool_size {
                println!("  -> Lane {}/{} loaded into VRAM", i + 1, pool_size);
            }
        }

        println!("--------------------------------------------------");
        println!("✅ POOL DEPLOYED: {} Parallel GPU Lanes Active", pool_size);
        println!("--------------------------------------------------");

        Self {
            pool: Arc::new(pool),
            debug_capture_count: Arc::new(AtomicUsize::new(0)),
            pool_size,
            total_vram_mb,
            active_streams: Arc::new(AtomicUsize::new(0)),
            stream_rois: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn calculate_pool_size(
        device: &nvml_wrapper::Device,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let name = device.name()?;
        let mem = device.memory_info()?;

        let free_mb = mem.free / 1024 / 1024;
        let total_mb = mem.total / 1024 / 1024;

        println!(
            "[GPU Check] Found: {} (Total: {}MB, Free: {}MB)",
            name, total_mb, free_mb
        );

        // Configuration:
        // Reserved VRAM for OS/Display: 1.5 GB (Safe margin for Windows)
        // Average YOLOv8n footprint (weights + TRT overhead): ~250 MB
        let reserved = 1536;
        let per_session = 250;

        if free_mb < reserved + per_session {
            return Ok(1);
        }

        let available = free_mb - reserved;
        let count = (available / per_session) as usize;

        // Target high parallelism for 12 cameras
        Ok(count.min(32).max(1))
    }

    fn create_session() -> Session {
        let builder = Session::builder()
            .expect("Failed to create SessionBuilder")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("Failed to set optimization level")
            .with_memory_pattern(true)
            .expect("Failed to set memory pattern")
            .with_config_entry("session.disable_cpu_ep_fallback", "1")
            .expect("Failed to disable CPU fallback");

        let trt_provider = TensorRTExecutionProvider::default()
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path("./assets/cache");

        let cuda_provider = CUDAExecutionProvider::default();

        builder
            .with_execution_providers([trt_provider.build(), cuda_provider.build()])
            .expect("GPU ENGAGEMENT FAILURE: DLLs missing or incompatible drivers.")
            .commit_from_file("assets/yolov8n.onnx")
            .expect("Failed to load model from assets/yolov8n.onnx")
    }
}
