use oxide_framework_core::{controller, ApiResponse, AppState, Data, Json};
use serde::Deserialize;
use crate::state::WorkerState;
use crate::streams::StreamManager;

#[derive(Deserialize)]
pub struct StreamAssignPayload {
    pub stream_id: String,
    pub url: String,
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
        let manager = StreamManager::new((*state).clone());
        manager.spawn_stream(payload.stream_id, payload.url).await;
        ApiResponse::ok(())
    }
}
