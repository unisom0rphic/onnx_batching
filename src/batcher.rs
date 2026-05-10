use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use tokio::{select, sync::mpsc::Receiver, task::spawn_blocking};

use log::debug;

use crate::onnx::OnnxModel;
use crate::web::InferenceRequest;

// the core logic is it selects on global channel, gets the InferenceRequest structure containing
// both the inputs and response_tx oneshot::sender,
// then waits either for timeout or for big enough batch_size and runs inference for every request

// It processes inputs to outputs and sends them with response_tx
// which is bound to response_rx inside of infer instance, where the response_rx
// is waiting for the results.

// TODO: graceful shutdown

pub struct Batcher {
    model: Arc<Mutex<OnnxModel>>,
}

impl Batcher {
    /// Initializes model from path
    pub fn init(path: &str) -> Self {
        let model = OnnxModel::load_onnx(path).expect("Model couldn't be loaded");
        Self {
            model: Arc::new(Mutex::new(model)),
        }
    }

    pub async fn run(
        &self,
        // TODO: global_ch_rx is impossible to pass because we only have the sender rn
        mut global_ch_rx: Receiver<InferenceRequest>,
        batch_size: usize,
        timeout_duration: Duration,
    ) {
        let mut buffer = Vec::with_capacity(batch_size);
        let mut timeout = tokio::time::sleep(timeout_duration);
        loop {
            select! {
                biased;
                Some(inf_req) = global_ch_rx.recv() => {
                    debug!("Received data from global_ch_rx: {:?}", inf_req);
                    buffer.push(inf_req);
                    if buffer.len() >= batch_size {
                        debug!("Send batch to processing (enough batch_size)");
                        let batch = std::mem::take(&mut buffer);
                        let model_clone = self.model.clone();
                        let _ = spawn_blocking(
                            move || {
                                // FIXME: handle unwrap()
                                let mut model = model_clone.lock().unwrap();
                                // need to extract data from requests
                                let _ = model.batch_infer(batch);
                            }
                        ).await;
                    }
                    timeout = tokio::time::sleep(timeout_duration);
                }
                _ = timeout => {
                    if !buffer.is_empty() {
                        debug!("Send batch to processing (timeout)");
                        let batch = std::mem::take(&mut buffer);
                        let model_clone = self.model.clone();
                        // same issues here, look above for information
                        let _ = spawn_blocking(
                            move || {
                                let mut model = model_clone.lock().unwrap();
                                let error = model.batch_infer(batch);
                                debug!("{:?}", error);
                            }
                        ).await;
                    }
                    timeout = tokio::time::sleep(timeout_duration);
                }
            }
        }
    }
}
