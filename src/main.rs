mod batcher;
mod onnx;
mod web;

fn main() {
    // WIP
    let path = "path";
    let model = onnx::OnnxModel::load_onnx(path).expect("Failed to load lol");
}
