# Batch Inference Server for ONNX
**An asynchronous, high-throughput educational project written in Rust**  
Implements dynamic request batching: multiple incoming requests are collected into a single   
batch, run through the model only once. The design significantly reduces per-request overhead  
for models like LLMs or heavy CV.  

### Features
- uses `tokio` to handle requests
- timeout-aware - batches are flushed when full or after a configurable idle timeout, preventing excessive latency
- `axum` HTTP server - simple POST endpoint expecting JSON-arrays
- ONNX runtime powered by `ort`
- backpressure! (bounded `tokio::sync::mpsc` channel)
- per-request `tokio::sync::oneshot` channels to ensure the right response order

### Technological stack
- `tokio`
- `axum`
- `ort`


### Architecture
```
Client ──POST /predict──► Axum handler
                              │
                     [InferenceRequest]
                              │
                              ▼
                     Global MPSC queue
                     (LazyLock<Sender>)
                              │
                              ▼
                     Batcher task (background)
                        │            │
                        └─ collects requests
                        │            │
                        └─ batch size reached OR timeout
                                   │
                                   ▼
                              Batch inference
                              (model runs once)
                                   │
                                   ▼
                         Zipped results → each oneshot::Sender
                                   │
                                   ▼
                              Client receives response
```

### Prerequisites
- Rust (used 1.93 for development)
- ONNX model (`.onnx` file)

### Setup
1. Clone the repository
```bash
git clone https://github.com/unisom0rphic/onnx_batching
cd onnx_batching
```
2. Place your ONNX model inside the `.src` folder  
3. Update the model path in code  
4. Build and run  
```bash
cargo build --release
cargo run --release
```

**The server will start at http://0.0.0.0:3000**  

### Running inference
`POST /predict`  
*Request body:* JSON array of floats (input features)  
`[0.1, 0.2, 0.3, ...]`  
*Response:* JSON array of floats  
`[0.7, ...]`

**Status codes:**  
- `200 OK`  
- `503 Service Unavailable` - batch queue is full  
- `500 Internal Server Error` - inference failed  

**Example run**  
```bash
# Terminal 1: start server
cargo run --release

# Terminal 2: send a request
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '[0.1, 0.2, 0.3, 0.4]'
```

### License  
[MIT](license.md)

