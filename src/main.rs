mod batcher;
mod onnx;
mod web;

fn main() {
    // WIP
    let path = "path";
    let _model = onnx::OnnxModel::load_onnx(path).expect("Failed to load lol");
}
