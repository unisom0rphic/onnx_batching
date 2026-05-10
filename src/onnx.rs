use log::debug;
use ort::{session::Session, value::Tensor};
use tokio::sync::oneshot;

use crate::web::InferenceRequest;

pub struct OnnxModel {
    session: Session,
}

static ONNX_INIT: std::sync::Once = std::sync::Once::new();

impl OnnxModel {
    // TODO: with_intra_threads for multithreading
    /// Loads specified ONNX model from PATH
    pub fn load_onnx(path: &str) -> ort::Result<Self> {
        debug!("Loading model from {}", path);
        ONNX_INIT.call_once(|| {
            ort::init().commit();
        });

        let session = Session::builder()?.commit_from_file(path)?;

        Ok(Self { session })
    }

    /// Runs batched inference for the provided session
    pub fn batch_infer(&mut self, request_vec: Vec<InferenceRequest>) -> ort::Result<()> {
        let (inputs_vec, senders_vec): (Vec<Vec<f32>>, Vec<oneshot::Sender<Vec<f32>>>) =
            request_vec
                .into_iter()
                .map(|r| (r.inputs, r.response_tx))
                .unzip();

        debug!("ONNX model inference called");
        debug!("Inputs: {:?}", inputs_vec);

        let rows = inputs_vec.len();
        if rows == 0 {
            return Ok(());
        }
        let columns = inputs_vec[0].len();

        let inputs = ndarray::Array2::<f32>::from_shape_vec(
            (rows, columns),
            inputs_vec.into_iter().flatten().collect(),
        )
        .map_err(|e| ort::Error::new(e.to_string()))?;

        let inputs = Tensor::from_array(inputs)?;

        // https://docs.rs/ort/latest/ort/session/struct.SessionOutputs.html
        let outputs = self.session.run(ort::inputs!["input" => inputs])?;

        // outputs is basically a hashmap {output: value} but as a tuple
        let value = &outputs["output"];
        let (shape, tensor_data) = value.try_extract_tensor::<f32>()?;

        let out_cols = shape[1] as usize;

        let results: Vec<_> = tensor_data
            .chunks_exact(out_cols)
            .map(|chunk| chunk.to_vec())
            .collect();

        for (sender, result) in senders_vec.into_iter().zip(results) {
            let _ = sender.send(result);
        }

        Ok(())
    }
}
