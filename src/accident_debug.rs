// ============================================================================
//  accident_debug.rs — Persists confirmed accident frames + metadata to disk.
//
//  Directory layout (relative to the worker's CWD):
//      debug/accidents/
//          <camera>_<unix_ms>.jpg     ← 640×640 JPEG frame at detection time
//          <camera>_<unix_ms>.json    ← accident report + vehicle/congestion context
//
//  All writes are best-effort; failures are logged but never propagate. Disable by
//  setting ACCIDENT_DEBUG=0 in the environment.
// ============================================================================

use image::{ImageOutputFormat, RgbImage};
use std::io::Cursor;
use std::path::PathBuf;
use tracing::{debug, warn};

use crate::alpen_detect::AccidentReport;

fn is_enabled() -> bool {
    !matches!(
        std::env::var("ACCIDENT_DEBUG")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Persist a confirmed accident snapshot to `debug/accidents/YYYY-MM-DD/`.
/// `context` is attached as JSON next to the JPEG so you can correlate vehicle
/// counts / congestion levels with the frame.
pub fn save_accident_snapshot(
    camera_name: &str,
    frame: &RgbImage,
    report: &AccidentReport,
    context: &serde_json::Value,
) {
    if !is_enabled() {
        return;
    }

    let base = std::env::var("ACCIDENT_DEBUG_DIR")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("debug").join("accidents"));

    if let Err(e) = std::fs::create_dir_all(&base) {
        warn!(
            "[{}] Failed to create accident debug dir {}: {}",
            camera_name,
            base.display(),
            e
        );
        return;
    }

    let unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let stem = format!("{}_{}", sanitize(camera_name), unix_ms);

    // JPEG frame
    let jpg_path = base.join(format!("{}.jpg", stem));
    let mut buf = Cursor::new(Vec::new());
    match frame.write_to(&mut buf, ImageOutputFormat::Jpeg(90)) {
        Ok(_) => {
            if let Err(e) = std::fs::write(&jpg_path, buf.into_inner()) {
                warn!(
                    "[{}] Failed writing accident JPEG {}: {}",
                    camera_name,
                    jpg_path.display(),
                    e
                );
            } else {
                debug!(
                    "[{}] Saved accident debug frame -> {}",
                    camera_name,
                    jpg_path.display()
                );
            }
        }
        Err(e) => {
            warn!("[{}] Failed encoding accident JPEG: {}", camera_name, e);
        }
    }

    // Sidecar JSON
    let json_path = base.join(format!("{}.json", stem));
    let sidecar = serde_json::json!({
        "camera_name": camera_name,
        "saved_at_unix_ms": unix_ms,
        "report": report,
        "context": context,
    });
    match serde_json::to_vec_pretty(&sidecar) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&json_path, bytes) {
                warn!(
                    "[{}] Failed writing accident sidecar {}: {}",
                    camera_name,
                    json_path.display(),
                    e
                );
            }
        }
        Err(e) => {
            warn!("[{}] Failed serializing accident sidecar: {}", camera_name, e);
        }
    }
}
