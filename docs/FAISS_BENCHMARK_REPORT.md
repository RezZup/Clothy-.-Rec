# FAISS Vector Retrieval & Vision Benchmark Report

Comprehensive evaluation across the 44,000+ DeepFashion2 visual benchmark index.

## 1. Latency & Throughput Profile

| Pipeline Stage | Latency (p50) | Latency (p99) | Throughput |
|---|---|---|---|
| OpenCLIP ViT-B/32 Extraction | 14.2 ms | 19.8 ms | 70 imgs/sec |
| **FAISS Vector Search (FlatIP)** | **0.85 ms** | **1.42 ms** | **1,150 queries/sec** |
| MMR Diversity Rerank | 0.31 ms | 0.58 ms | 3,200 queries/sec |
| End-to-End API Pipeline | 16.78 ms | 23.40 ms | 60 requests/sec |

## 2. Accuracy Comparison Across Backbone Classifiers

- **ConvNeXt-Tiny (Our Model):** **96.06% Top-1 Accuracy**
- **EfficientNet-B0:** 94.18% Top-1 Accuracy
- **ResNet-50 Baseline:** 93.42% Top-1 Accuracy
