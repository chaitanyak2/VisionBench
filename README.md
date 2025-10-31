# VisionSearch Engine

![Code Lines](https://img.shields.io/badge/code%20lines-350-blue)
![Test Lines](https://img.shields.io/badge/test%20lines-135-green)

---

## 🧭 Overview

**VisionSearch** is a modular, benchmark-aware **visual similarity search engine** built on  
**GStreamer**, **LibTorch**, **SQLite**, and **FAISS**.  
It enables both **image ingestion** and **retrieval/search** pipelines that can be monitored and benchmarked in real time.

The framework follows a clean layered architecture to allow fast experimentation with new preprocessing, inference, and storage components.

---

## 🚀 Current Progress

| Component | Status | Notes |
|------------|:------:|-------|
| **Core Metadata** | ✅ Done | Unified metadata for all buffers and pipeline stages |
| **BenchmarkRecorder** | ✅ Done | Asynchronous SQLite-based benchmark logger |
| **GStreamer Integration** | ⏳ Next | Custom `GstMeta` for CoreMetadata |
| **VectorDB (FAISS)** | ⏳ Next | Embedding storage + ID mapping |
| **Ingest Pipeline** | 🔜 Planned | Source → Preprocess → Infer → VectorDB Sink |
| **Retrieve Pipeline** | 🔜 Planned | Query Source → Infer → VectorDB Search → Result Sink |

---

## 🧱 Proposed Layered Architecture



# VisionBench
**VisionBench**  is a modular C++ benchmarking framework for evaluating and comparing visual embedding models, indexing strategies, and similarity search pipelines. It runs in benchmark mode to collect detailed performance, accuracy, and system metrics—storing all experiment results as structured JSON and SQLite records for reproducible analysis.It is built on top of **GStreamer** / **DeepStream**.

### Features
- GStreamer-based ingest and inference pipelines 
- Benchmark and collect latency, recall, throughput metrics 
- Store experiments in SQLite / JSON 
- Extensible plugin design (`vbinfer`, `vbindex`, `vbmetrics`) 
- Supports ONNXRuntime (CPU) and TensorRT (GPU) backends 

## 🛣️ Roadmap: *First Working Ingest + Retrieve Pipeline*

| Milestone | Description | Status |
|------------|--------------|:------:|
| **M1** | CoreMetadata and BenchmarkRecorder | ✅ Done |
| **M2** | GStreamer Metadata integration (`GstMeta` wrapper) | ⏳ In Progress |
| **M3** | VectorDBClient (FAISS + SQLite map) | 🔜 Planned |
| **M4** | Ingest Pipeline prototype (CPU-only, ResNet-50) | 🔜 Planned |
| **M5** | Retrieve Pipeline (search by image embedding) | 🔜 Planned |
| **M6** | Unified Application Layer to orchestrate both | 🔜 Planned |
| **M7** | End-to-end benchmark and visualization dashboard | 🧭 Future |

---

## 🧩 Architectural Notes

- **Asynchronous Benchmarking:**  
  Every plugin reports performance metrics via a shared `BenchmarkRecorder`,  
  enabling non-blocking metric logging while pipelines run concurrently.

- **Metadata-Driven Flow:**  
  Each buffer carries a `CoreMetadata` object that tracks IDs, stage, and embedding pointers,  
  allowing seamless coordination across elements.

- **Two Databases:**  
  - **SQLite:** lightweight structured logging (benchmark + metadata).  
  - **FAISS:** fast high-dimensional vector search for embeddings.


---

## 🧪 Development and Testing

Build locally:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

---

## Run tests manually:

> LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./tests/test_benchmark

---

## Docker Build
```bash
docker build -t vis-core:latest .
docker run --rm -it vis-core:latest
```

🧠 Future Extensions

GPU / CUDA inference pipeline

Distributed FAISS or Milvus backend

Real-time ingestion from video streams

Web UI for visualization and benchmark dashboards

Integration with Grafana / Prometheus metrics

---

Current milestone: ✅ Core components established (BenchmarkRecorder, Metadata).
Next goal: integrate GStreamer metadata and build the first Ingest + Retrieve pipelines.

