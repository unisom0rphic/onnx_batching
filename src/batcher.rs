use ort::session::SessionOutputs;
use std::{sync::Arc, time::Duration};
use tokio::{
    select,
    sync::{
        mpsc::{Receiver, Sender},
        oneshot,
    },
    task,
};

use crate::onnx::OnnxModel;

use crate::web::InferenceRequest;

// the core logic is it selects on global channel, gets the InferenceRequest structure containing
// both the inputs and response_tx oneshot::sender,
// then waits either for timeout or for big enough batch_size and runs inference for every request

// It processes inputs to outputs and sends them with response_tx
// which is bound to response_rx inside of infer instance, where the response_rx
// is waiting for the results.

struct Batcher {}

impl Batcher {}

// global_ch_rx is impossible to pass because we only have the sender rn
async fn run(
    mut global_ch_rx: Receiver<InferenceRequest>,
    batch_size: usize,
    timeout_duration: Duration,
) {
    let mut buffer = Vec::with_capacity(batch_size);
    let mut timeout = tokio::time::sleep(timeout_duration);
    loop {
        select! {
            biased;
            inf_req = global_ch_rx.recv() => {
                buffer.push(inf_req.unwrap()); // TODO: handle None properly
                if buffer.len() >= batch_size {
                    let batch = std::mem::take(&mut buffer); // takes ownership of the Vec contents
                    infer_batch(batch).await;                // batch is moved in, buffer is now empty
                }
                timeout = tokio::time::sleep(timeout_duration);
            }
            _ = timeout => {
                if !buffer.is_empty() {
                    let batch = std::mem::take(&mut buffer);
                    infer_batch(batch).await;
                }
                timeout = tokio::time::sleep(timeout_duration);
            }
        }
    }
}

// NOTE: using &Vec<T> instead of &[T] is an antipattern because
// Vec<T> is *guaranteed* to be *contiguous* in memory;
// Vec<T> is already a pointer to heap-allocated memory
// If the function receives an array, it needs to allocate new Vec
async fn infer_batch(batch: Vec<InferenceRequest>) {
    // take ownership, not &[]
    let inputs: Vec<Vec<f32>> = batch.iter().map(|req| req.inputs.clone()).collect();
    // Run batch inference once (e.g., model.run(&inputs))
    let outputs = run_batch_inference(&inputs).await;
    for (req, output) in batch.into_iter().zip(outputs) {
        let _ = req.response_tx.send(output);
    }
}

async fn run_batch_inference(inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    inputs.to_vec()
}
