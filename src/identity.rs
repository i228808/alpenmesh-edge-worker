use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerIdentity {
    pub worker_id: String,
    pub worker_secret: String,
    pub created_at_ms: u64,
}

pub fn load_or_create_identity() -> io::Result<WorkerIdentity> {
    let path = identity_file_path()?;
    if path.exists() {
        let raw = fs::read_to_string(&path)?;
        let parsed: WorkerIdentity = serde_json::from_str(&raw).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Failed parsing worker identity JSON: {}", e),
            )
        })?;
        return Ok(parsed);
    }

    let identity = WorkerIdentity {
        worker_id: format!("wrk_{}", Uuid::new_v4().simple()),
        worker_secret: format!("sec_{}", Uuid::new_v4().simple()),
        created_at_ms: now_unix_ms(),
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(&identity).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed serializing worker identity: {}", e),
        )
    })?;
    fs::write(&path, payload)?;
    Ok(identity)
}

pub fn identity_file_path() -> io::Result<PathBuf> {
    if let Ok(custom) = std::env::var("WORKER_IDENTITY_PATH") {
        return Ok(PathBuf::from(custom));
    }

    if cfg!(target_os = "windows") {
        if let Ok(app_data) = std::env::var("APPDATA") {
            return Ok(Path::new(&app_data)
                .join("alpenmesh")
                .join("worker_identity.json"));
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        return Ok(Path::new(&home)
            .join(".config")
            .join("alpenmesh")
            .join("worker_identity.json"));
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "Unable to resolve home directory for worker identity",
    ))
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
