use crate::alpen_detect::ALPEN_DETECT_MODEL_PATH;
use crate::inference::RoiMap;
use crate::night_enhance::EnlightenGanSession;
use crate::tracker::Tracker;
use crossbeam_queue::ArrayQueue;
use nvml_wrapper::Nvml;
use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;
use std::sync::RwLock;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

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
    /// Active stream tasks currently owned by this worker
    pub running_streams: Arc<RwLock<HashSet<String>>>,
    /// Batched proof-of-inference counters flushed periodically to scheduler
    pub proof_accumulator: Arc<RwLock<ProofAccumulator>>,
    /// Stable worker identity used in proof batching and scheduler accounting
    pub worker_id: Arc<String>,
    /// Local persistent secret for secure worker registry binding
    pub worker_secret: Arc<String>,
    /// Externally reachable address of this worker's HTTP server ("host:port").
    /// Sent in heartbeats so the scheduler can post stream assignments back.
    pub external_addr: Arc<String>,
    /// Base URL of the scheduler (no trailing slash), used for all upstream posts.
    pub scheduler_url: Arc<String>,
    /// Base URL of the economy service, used for identity registration.
    pub economy_url: Arc<String>,
    /// Optional night enhancement stage.
    pub night_enhancer: Arc<EnlightenGanSession>,
    /// Tracks cooldowns for High Congestion alerts (used by stateless /execute path).
    /// The live stream path uses PerStreamState.congestion_cooldown_until instead.
    pub congestion_cooldowns: Arc<RwLock<HashMap<String, f64>>>,
    /// Per-stream state: tracker, motion hashes, cooldowns (live stream path).
    /// Outer RwLock for stream registration; inner Mutex so one slow stream
    /// doesn't block others.
    pub per_stream: Arc<RwLock<HashMap<String, Arc<Mutex<PerStreamState>>>>>,
}

/// Per-stream state for the live RTSP ingestion path.
/// Holds the IoU tracker, motion pre-filter hashes, and congestion cooldown.
pub struct PerStreamState {
    pub tracker: Tracker,
    pub last_roi_hashes: HashMap<String, [u8; 64]>,
    pub last_frame_full_hash: Option<[u8; 64]>,
    pub congestion_cooldown_until: f64,
    pub last_frame_idx: u64,
}

#[derive(Clone, Debug)]
pub struct ProofAccumulator {
    pub window_start_ms: u64,
    pub total_inferences: u64,
    pub camera_inferences: HashMap<String, u64>,
}

#[derive(Clone, Debug)]
pub struct ProofBatch {
    pub window_start_ms: u64,
    pub window_end_ms: u64,
    pub total_inferences: u64,
    pub camera_inferences: HashMap<String, u64>,
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
            let mut session = Self::create_session();
            // Run a dummy inference so TensorRT compiles and caches its engine
            // now (at startup) rather than on the first real video frame.
            Self::warmup_session(&mut session);
            pool.push(Arc::new(Mutex::new(session)))
                .expect("Failed to fill session pool");
            if (i + 1) % 5 == 0 || i + 1 == pool_size {
                println!("  -> Lane {}/{} loaded into VRAM", i + 1, pool_size);
            }
        }

        println!("--------------------------------------------------");
        println!("✅ POOL DEPLOYED: {} Parallel GPU Lanes Active", pool_size);
        println!("--------------------------------------------------");

        let identity = crate::identity::load_or_create_identity()
            .expect("Failed to load/create local worker identity");

        let night_enhancer = EnlightenGanSession::new();

        // Externally reachable address for this worker. Defaults preserve previous
        // behavior but multi-worker setups should set WORKER_EXTERNAL_ADDR.
        let external_addr = std::env::var("WORKER_EXTERNAL_ADDR")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "127.0.0.1:8081".to_string());

        let scheduler_url = std::env::var("SCHEDULER_URL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "http://127.0.0.1:3000".to_string())
            .trim_end_matches('/')
            .to_string();

        let economy_url = std::env::var("ECONOMY_URL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "http://127.0.0.1:3010".to_string())
            .trim_end_matches('/')
            .to_string();

        Self {
            pool: Arc::new(pool),
            debug_capture_count: Arc::new(AtomicUsize::new(0)),
            pool_size,
            total_vram_mb,
            active_streams: Arc::new(AtomicUsize::new(0)),
            stream_rois: Arc::new(RwLock::new(HashMap::new())),
            running_streams: Arc::new(RwLock::new(HashSet::new())),
            proof_accumulator: Arc::new(RwLock::new(ProofAccumulator {
                window_start_ms: now_unix_ms(),
                total_inferences: 0,
                camera_inferences: HashMap::new(),
            })),
            worker_id: Arc::new(
                std::env::var("WORKER_ID")
                    .ok()
                    .filter(|v| !v.trim().is_empty())
                    .unwrap_or_else(|| identity.worker_id.clone()),
            ),
            worker_secret: Arc::new(identity.worker_secret),
            external_addr: Arc::new(external_addr),
            scheduler_url: Arc::new(scheduler_url),
            economy_url: Arc::new(economy_url),
            night_enhancer: Arc::new(night_enhancer),
            congestion_cooldowns: Arc::new(RwLock::new(HashMap::new())),
            per_stream: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn record_inference(&self, camera_name: &str) {
        let mut guard = self.proof_accumulator.write().unwrap();
        guard.total_inferences = guard.total_inferences.saturating_add(1);
        let entry = guard
            .camera_inferences
            .entry(camera_name.to_string())
            .or_insert(0);
        *entry = entry.saturating_add(1);
    }

    pub fn drain_proof_batch(&self) -> Option<ProofBatch> {
        let now_ms = now_unix_ms();
        let mut guard = self.proof_accumulator.write().unwrap();

        if guard.total_inferences == 0 {
            if now_ms.saturating_sub(guard.window_start_ms) >= 30_000 {
                guard.window_start_ms = now_ms;
            }
            return None;
        }

        let batch = ProofBatch {
            window_start_ms: guard.window_start_ms,
            window_end_ms: now_ms,
            total_inferences: guard.total_inferences,
            camera_inferences: guard.camera_inferences.clone(),
        };

        guard.window_start_ms = now_ms;
        guard.total_inferences = 0;
        guard.camera_inferences.clear();

        Some(batch)
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
        // Reserved VRAM for OS/Display: 1 GB
        // YOLOv11 TensorRT (FP16) lanes actually take less VRAM than the old model.
        let reserved = 1024;
        let per_session = 350;

        if free_mb < reserved + per_session {
            return Ok(1);
        }

        let available = free_mb - reserved;
        let count = (available / per_session) as usize;

        // Target high parallelism for 12 cameras
        Ok(count.min(32).max(1))
    }

    /// Drive a blank 640×640 frame through the session so TensorRT compiles and
    /// caches its engine immediately.  This trades a few seconds at startup for
    /// near-zero latency on the first real frame.
    fn warmup_session(session: &mut Session) {
        let blank = vec![0u8; 640 * 640 * 3];
        let _ = crate::alpen_detect::run_session(session, &blank);
    }

    fn create_session() -> Session {
        let builder = Session::builder()
            .expect("Failed to create SessionBuilder")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("Failed to set optimization level")
            .with_memory_pattern(true)
            .expect("Failed to set memory pattern");

        let trt_provider = TensorRTExecutionProvider::default()
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path("./assets/cache");

        let cuda_provider = CUDAExecutionProvider::default();

        builder
            .with_execution_providers([trt_provider.build(), cuda_provider.build()])
            .expect("GPU ENGAGEMENT FAILURE: DLLs missing or incompatible drivers.")
            .commit_from_file(ALPEN_DETECT_MODEL_PATH)
            .expect("Failed to load model from assets/alpen_detect.onnx")
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
