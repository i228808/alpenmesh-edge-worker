use serde::Serialize;

/// Report emitted when a tracked vehicle has been stationary in a through-lane
/// for longer than `ALPEN_STALL_THRESHOLD_SEC`.
///
/// Separate from `AccidentReport` — different semantics, different dispatch,
/// and different cooldown behaviour.
#[derive(Clone, Debug, Serialize)]
pub struct StalledVehicleReport {
    pub track_id: u32,
    pub location_bbox: [f32; 4],
    pub dwell_sec: f32,
    pub detected_at: f64,
    pub reason: String,
}

impl StalledVehicleReport {
    /// Build from a tracker stalled event.
    pub fn from_stall(track_id: u32, bbox: [f32; 4], dwell_sec: f32, timestamp: f64) -> Self {
        Self {
            track_id,
            location_bbox: bbox,
            dwell_sec,
            detected_at: timestamp,
            reason: format!(
                "Vehicle track {} stationary for {:.1}s (threshold: {}s)",
                track_id,
                dwell_sec,
                std::env::var("ALPEN_STALL_THRESHOLD_SEC")
                    .unwrap_or_else(|_| "90".to_string()),
            ),
        }
    }
}
