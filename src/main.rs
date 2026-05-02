mod onnx;
mod web;
mod batcher;

fn main() {
    // WIP
    let path = "path";
    let model = onnx::OnnxModel::load_onnx(path).expect("Failed to load lol");
}
