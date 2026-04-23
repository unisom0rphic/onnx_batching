use ort::session::SessionOutputs;
use tokio::sync::{mpsc::{Receiver, Sender}, oneshot};
use std::sync::Arc;

use crate::onnx::OnnxModel;

// the core logic is it selects on global channel, gets the InferenceRequest structure containing 
// both the inputs and response_tx oneshot::sender, 
// then waits either for timeout or for enough batchsize and runs inferences for every request.

// It processed inputs to outputs and sends them with response_tx 
// which is bound to response_rx inside of infer instance, where the response_rx 
// is waiting for the results.

struct Batcher {
    
}

impl Batcher {

}