use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::post,
};
use log::{debug, info};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
pub struct InferenceRequest {
    pub inputs: Vec<f32>,
    pub response_tx: oneshot::Sender<Vec<f32>>,
}

async fn send_infer_request(
    State(request_tx): State<mpsc::Sender<InferenceRequest>>,
    Json(inputs): Json<Vec<f32>>,
) -> Result<Json<Vec<f32>>, (StatusCode, String)> {
    info!("Received JSON for inference: {:?}", inputs);
    let (response_tx, response_rx) = oneshot::channel();
    let req = InferenceRequest {
        inputs,
        response_tx,
    };
    debug!("Sending data to global channel");
    request_tx.send(req).await.map_err(|e| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Batch queue full: {}", e),
        )
    })?;
    debug!("Waiting for results");
    let result = response_rx.await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Inference failed: {}", e),
        )
    })?;
    Ok(Json(result))
}

pub async fn run(request_tx: mpsc::Sender<InferenceRequest>) {
    const ADDRESS: &str = "0.0.0.0:3000";
    let app = Router::new()
        .route("/predict", post(send_infer_request))
        .with_state(request_tx);

    // TODO: remove unwraps
    let listener = tokio::net::TcpListener::bind(ADDRESS).await.unwrap();
    info!("Started HTTP server on {ADDRESS}");
    axum::serve(listener, app).await.unwrap();
}
