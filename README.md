# VisionBench
**VisionBench**  is a modular C++ benchmarking framework for evaluating and comparing visual embedding models, indexing strategies, and similarity search pipelines. It runs in benchmark mode to collect detailed performance, accuracy, and system metrics—storing all experiment results as structured JSON and SQLite records for reproducible analysis.It is built on top of **GStreamer** / **DeepStream**.

### Features
- GStreamer-based ingest and inference pipelines 
- Benchmark and collect latency, recall, throughput metrics 
- Store experiments in SQLite / JSON 
- Extensible plugin design (`vbinfer`, `vbindex`, `vbmetrics`) 
- Supports ONNXRuntime (CPU) and TensorRT (GPU) backends 

### Build
```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build

