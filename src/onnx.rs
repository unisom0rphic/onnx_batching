use ndarray::{Array, Array2, ArrayBase, Axis, Dim, ShapeError, ViewRepr};
use ort::{session::{Session, SessionOutputs}, value::{Tensor, TensorRef}};

pub struct OnnxModel {
    session: Session
}

impl OnnxModel {
    // TODO: with_intra_threads for multithreading
    /// Loads specified ONNX model from PATH
    pub fn load_onnx(path: &str) -> ort::Result<Self> {
        ort::init().commit();

        let session = Session::builder()?.commit_from_file(path)?;

        Ok( Self{session} )
    }

    /// Runs batched inference for the provided session
    pub fn batch_infer(&mut self, input_batch: Vec<Vec<f32>>) -> ort::Result<Vec<Vec<f32>>> {
        // should run batched inference for the input batch,
        // which is either a 2D vector or NDarray

        // outputs (self.session.run()) is SessionOutputs<'_>
        // and outputs[i].try_extract_tensor() returns (&Shape, &[f32])

        // https://docs.rs/ort/latest/ort/session/struct.SessionOutputs.html

        // we can *probably* iterate through outputs though not sure,
        // iirc it contains three fields which are keys, values and length?
        todo!("WIP");
    }
}