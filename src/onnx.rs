use ndarray::Array;
use ort::{session::{Session, SessionOutputs}, value::TensorRef};

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

    /// Runs inference for the provided session
    pub fn infer(&mut self, inputs: Vec<f32>) -> ort::Result<SessionOutputs<'_>> {
        let array = Array::from_vec(inputs);
        let outputs = self.session.run(ort::inputs![TensorRef::from_array_view(&array)?])?;
        Ok(outputs)
    }
}