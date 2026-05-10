// TODO: refactor web and main

mod batcher;
mod onnx;
mod web;

use std::{sync::LazyLock, time::Duration};

use tokio::sync::mpsc;

use axum::{Json, Router, http::StatusCode, routing::post};
use serde::Serialize;
use tokio::sync::oneshot;

use crate::batcher::Batcher;
use log::{debug, info};

use crate::web::InferenceRequest;

// #[derive(Debug)]
// pub struct InferenceRequest {
//     pub inputs: Vec<f32>,
//     pub response_tx: oneshot::Sender<Vec<f32>>,
// }

// #[derive(Serialize, Debug)]
// struct InferenceResponse {
//     outputs: Vec<f32>,
// }

static BATCHER: LazyLock<Batcher> = LazyLock::new(|| Batcher::init("model.onnx"));

static REQUEST_TX: LazyLock<mpsc::Sender<InferenceRequest>> = LazyLock::new(|| {
    let (tx, rx) = mpsc::channel::<InferenceRequest>(100);
    tokio::spawn(async move {
        BATCHER.run(rx, 8, Duration::from_secs(5)).await;
    });
    tx
});

async fn send_infer_request(
    Json(inputs): Json<Vec<f32>>,
) -> Result<Json<Vec<f32>>, (StatusCode, String)> {
    info!("Received JSON for inference: {:?}", inputs);
    let (response_tx, response_rx) = oneshot::channel();
    let req = InferenceRequest {
        inputs,
        response_tx,
    };
    debug!("Sending data to global channel");
    REQUEST_TX.send(req).await.map_err(|e| {
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

#[tokio::main]
async fn main() {
    const ADDRESS: &str = "0.0.0.0:3000";
    env_logger::init();
    let app = Router::new().route("/predict", post(send_infer_request));

    // TODO: remove unwraps
    let listener = tokio::net::TcpListener::bind(ADDRESS).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    info!("Started HTTP server on {ADDRESS}");
}
