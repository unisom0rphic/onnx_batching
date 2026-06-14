mod batcher;
mod onnx;
mod web;

use std::time::Duration;

use tokio::sync::mpsc;

use crate::batcher::Batcher;
use crate::web::InferenceRequest;

#[tokio::main]
async fn main() {
    env_logger::init();

    let batcher = Batcher::init("model.onnx");
    let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>(100);

    tokio::spawn(async move {
        batcher.run(request_rx, 8, Duration::from_secs(5)).await;
    });

    web::run(request_tx).await;
}
