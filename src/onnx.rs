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

    /// Runs inference for the provided session
    pub fn batch_infer(&mut self, input_batch: Vec<Vec<f32>>) -> ort::Result<Vec<Vec<f32>>> {
        let batch_size = input_batch.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }
        let feature_dim = input_batch[0].len();

        // Flatten input and create a 2D array
        let flat: Vec<f32> = input_batch.into_iter().flatten().collect();
        let array = Array2::from_shape_vec((batch_size, feature_dim), flat)
            .map_err(|e| ort::Error::new(e.to_string()))?;

        // Run the model
        let outputs = self
            .session
            .run(ort::inputs![TensorRef::from_array_view(&array)?])?;

        // Assume a single output – pick the first one
        let (_output_name, output_value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| ort::Error::new("Model produced no outputs"))?;

        // Extract an owned array (consumes output_value)
        let view = output_value.try_extract_array::<f32>()?;
        let data: ndarray::ArrayD<f32> =     view.to_owned();

        // Verify it's 2D
        let shape = data.shape().to_vec();
        if shape.len() != 2 {
            return Err(ort::Error::new(format!(
                "Expected output of rank 2, got shape {:?}",
                shape
            )));
        }

        // Convert to Vec<Vec<f32>> by iterating over rows
        let result: Vec<Vec<f32>> = data
            .to_shape((shape[0], shape[1]))
            .map_err(|e: ShapeError| ort::Error::new(e.to_string()))?
            .axis_iter(Axis(0))
            .map(|row: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>, f32>| row.to_vec())
            .collect();

        Ok(result)
    }
}