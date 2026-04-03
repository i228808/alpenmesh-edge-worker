use oxide_framework_core::{controller, ApiResponse, AppState, Data, Json};
use serde::Deserialize;
use crate::state::WorkerState;
use crate::streams::StreamManager;
use crate::inference::RoiMap;

#[derive(Deserialize)]
pub struct StreamAssignPayload {
    pub stream_id: String,
    pub camera_name: String,
    pub url: String,
    #[serde(default)]
    pub rois: RoiMap,
}

pub struct StreamController;

#[controller("/api/v1/streams")]
impl StreamController {
    fn new(_state: &AppState) -> Self {
        Self
    }

    #[post("/assign")]
    async fn assign(
        &self,
        Data(state): Data<WorkerState>,
        Json(payload): Json<StreamAssignPayload>,
    ) -> ApiResponse<()> {
        {
            let running = state.running_streams.read().unwrap();
            if running.contains(&payload.stream_id) {
                return ApiResponse::ok(());
            }
        }

        {
            let mut running = state.running_streams.write().unwrap();
            if running.contains(&payload.stream_id) {
                return ApiResponse::ok(());
            }
            running.insert(payload.stream_id.clone());
        }

        if !payload.rois.is_empty() {
            let mut rois = state.stream_rois.write().unwrap();
            rois.insert(payload.stream_id.clone(), payload.rois.clone());
            rois.insert(payload.camera_name.clone(), payload.rois.clone());
        }

        let manager = StreamManager::new((*state).clone());
        manager.spawn_stream(payload.stream_id, payload.url).await;
        ApiResponse::ok(())
    }
}
