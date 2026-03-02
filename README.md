<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/ITcarrot/RoundPipe/raw/main/docs/assets/banner.dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/ITcarrot/RoundPipe/raw/main/docs/assets/banner.light.svg">
    <img src="https://github.com/ITcarrot/RoundPipe/raw/main/docs/assets/banner.light.svg" alt="RoundPipe Banner" width="400">
  </picture>
</p>

<p align="center">High Performance · Easy to Use · Built for Gaming GPUs</p>

<p align="center">
  <a href="https://pypi.org/project/roundpipe/"><img src="https://img.shields.io/pypi/v/roundpipe.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/roundpipe/"><img src="https://img.shields.io/pypi/pyversions/roundpipe.svg" alt="Python"></a>
  <a href="https://github.com/ITcarrot/RoundPipe/blob/main/COPYING.LESSER"><img src="https://img.shields.io/badge/license-LGPL--3.0-blue.svg" alt="License"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://clang.llvm.org/docs/ClangFormat.html"><img src="https://img.shields.io/badge/code%20style-clang--format-blue.svg" alt="Code style: clang-format"></a>
</p>

<p align="center">
  <a href="https://itcarrot.github.io/RoundPipe/">Documentation</a> ·
  <a href="https://itcarrot.github.io/RoundPipe/zh">中文文档</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="https://github.com/ITcarrot/RoundPipe/tree/main/example">Examples</a>
</p>

---

RoundPipe is a large DNN training framework that lets you train huge models on consumer-grade GPUs. On a single 24 GB GPU, you can full fine-tune 32B-parameter models, LoRA fine-tune up to 235B, and handle 64K+ token sequences, with throughput approaching datacenter-class hardware.

## Highlights

- **Train bigger than ever**: Full fine-tune 32B models or LoRA fine-tune up to 235B on a single 24 GB GPU. Up to 7× longer sequence length than PyTorch FSDP.
- **High performance**: Push a 4090 close to A800 NVLINK-class throughput. Up to 6× faster than FSDP Offload in typical workloads.
- **Linear multi-GPU scaling**: Scale to multiple GPUs within a node without rewriting your training loop. Throughput grows linearly while max sequence length per GPU stays unchanged.
- **Feels like PyTorch**: Sequential programming interface with a low learning curve. Works well in Jupyter Notebook for rapid iteration.
- **General by design**: No constraints on layer structure, training flow, or parameter update strategy.
- **Portable across accelerators**: Pure PyTorch implementation. Runs on Nvidia, AMD, and Ascend platforms.

## Benchmarks

All benchmarks below are measured on a single node with 8 GPUs. "OOM" means the framework cannot fit the model under that configuration.

### Maximum Input Sequence Length

| Framework | Qwen3-1.7B | Llama3.1-8B | Qwen3-32B | Qwen3-235B (LoRA) |
|---|---:|---:|---:|---:|
| 4090 · FSDP Offload | 11 K | 11 K | OOM | OOM |
| 4090 · **RoundPipe** | **73 K** | **49 K** | **28 K** | **31 K** |
| A800 · FSDP | 39 K | 29 K | 11 K | OOM |
| A800 · **RoundPipe** | **288 K** | **226 K** | **126 K** | **118 K** |

### Training Throughput (tokens/s)

| Framework | Qwen3-1.7B | Llama3.1-8B | Qwen3-32B | Qwen3-235B (LoRA) |
|---|---:|---:|---:|---:|
| 4090 · FSDP Offload | 35,074 | 4,071 | OOM | OOM |
| 4090 · **RoundPipe** | **65,417** | **24,275** | **5,516** | **1,820** |
| A800 · FSDP | 85,829 | 29,148 | 3,455 | OOM |
| A800 · **RoundPipe** | **84,692** | **28,427** | **6,301** | **1,796** |

### Multi-GPU Scaling (8× RTX 4090)

| GPUs | Qwen3-1.7B | Llama3.1-8B | Qwen3-32B | Qwen3-235B (LoRA) |
|---:|---:|---:|---:|---:|
| 1 | 8,881 | 3,142 | 740 | 480 |
| 2 | 17,026 | 6,259 | 1,476 | 808 |
| 4 | 33,178 | 12,278 | 2,897 | 1,281 |
| 8 | 65,417 | 24,275 | 5,516 | 1,820 |

Max sequence length per GPU stays constant across all GPU counts (73 K, 49 K, 28 K, and 31 K respectively).

### Cross-Platform

| Device | Qwen3-1.7B | Llama3.1-8B | Qwen3-32B | Qwen3-235B (LoRA) |
|---|---:|---:|---:|---:|
| AMD W7800 | 17,852 | 5,915 | 1,450 | 665 |
| Ascend 910B | 50,599 | 23,253 | 5,028 | 459 |
| RTX 4090 | 65,417 | 24,275 | 5,516 | 1,820 |

## Quick Start

### Installation

```bash
pip install roundpipe
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.4

### Examples

See the [`example/`](https://github.com/ITcarrot/RoundPipe/tree/main/example) directory. More examples and tutorials will be added soon.

## Documentation

Full documentation is available at **[itcarrot.github.io/RoundPipe](https://itcarrot.github.io/RoundPipe/)**.

中文文档请访问 **[itcarrot.github.io/RoundPipe/index.zh.html](https://itcarrot.github.io/RoundPipe/zh)**。

## License

RoundPipe is licensed under the [LGPL-3.0](COPYING.LESSER).
