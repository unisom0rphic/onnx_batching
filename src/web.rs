use std::sync::LazyLock;

use tokio::sync::mpsc;

use axum::{Json, Router, http::StatusCode, routing::post};
use serde::Serialize;
use tokio::sync::oneshot;

pub struct InferenceRequest {
    pub inputs: Vec<f32>,
    pub response_tx: oneshot::Sender<Vec<f32>>,
}

#[derive(Serialize)]
struct InferenceResponse {
    outputs: Vec<f32>,
}

static REQUEST_TX: LazyLock<mpsc::Sender<InferenceRequest>> = LazyLock::new(|| {
    let (tx, mut rx) = mpsc::channel::<InferenceRequest>(100);
    tokio::spawn(async move {
        // infinite loop waiting for rx
        while let Some(req) = rx.recv().await {
            // replace with batcher.run(req)
            let result = vec![];
            let _ = req.response_tx.send(result);
        }
    });
    tx
});

async fn send_infer_request(
    Json(inputs): Json<Vec<f32>>,
) -> Result<Json<Vec<f32>>, (StatusCode, String)> {
    let (response_tx, response_rx) = oneshot::channel();
    let req = InferenceRequest {
        inputs,
        response_tx,
    };
    REQUEST_TX.send(req).await.map_err(|e| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Batch queue full: {}", e),
        )
    })?;
    let result = response_rx.await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Inference failed: {}", e),
        )
    })?;
    Ok(Json(result))
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/predict", post(send_infer_request));

    // TODO: remove unwraps
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
