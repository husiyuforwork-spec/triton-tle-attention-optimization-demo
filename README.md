<div align="center">
  <h1>Triton-TLE Attention Optimization Demo</h1>
  <p>
    <img alt="GPU" src="https://img.shields.io/badge/GPU-NVIDIA%20H100%20%7C%20SM90-76B900">
    <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.8%2B-76B900">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-3776AB">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
  </p>
</div>

**中文说明：** [README_cn.md](README_cn.md)

This repository accompanies the technical presentation **“Hands-on Optimization of a New Attention Kernel with Triton-TLE”**. It is intended for technical exchange, live demonstrations, and experiment reproduction. The project presents two data-layout optimization implementations for an Attention kernel on Hopper GPUs and includes the corresponding Nsight Compute reports.

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#implementation-comparison">Implementation Comparison</a> |
  <a href="#performance-materials">Performance Materials</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#reproduce-ncu-reports">Reproduce NCU Reports</a>
</p>

## Overview

The demo compares the pre-optimization RS-WGMMA implementation with the optimized Transposed-Memdesc implementation. RS-WGMMA loads the V tile into registers and performs an explicit register-level transpose before register-source WGMMA, while Transposed-Memdesc expresses the transposed layout through a memory descriptor and reduces the shared-memory transpose traffic before issuing the same WGMMA computation. It includes source code, independent benchmark entry points, pinned experiment dependencies, and `.ncu-rep` reports for post-presentation review and study.

## Implementation Comparison

| Implementation | Data-layout handling | Compute path |
|---|---|---|
| RS-WGMMA | Register-level data transpose | WGMMA register-source |
| Transposed-Memdesc | Transposed memory descriptor | WGMMA register-source |

Both implementations use the same `uniform_4096` workload, enabling direct comparison of Kernel Duration, instruction counts, shared-memory traffic, bank conflicts, occupancy, and memory throughput in Nsight Compute.

## Workload Configuration

The demo uses the following fixed `uniform_4096` KV-cache configuration:

| Parameter | Value |
|---|---:|
| Batch size | 64 sequences |
| KV-cache length | 4096 tokens per sequence |
| Query tokens per decode step | 1 |
| Query heads | 8 |
| KV heads | 1 |
| Head dimension | 128 |

## Performance Materials

| Benchmark | Kernel | Report |
|---|---|---|
| `benchmark/bench_rswgmma.py` | `rswgmma_uniform4096_kernel` | `reports/rswgmma_uniform4096.ncu-rep` |
| `benchmark/bench_memdesc_transpose.py` | `memdesc_transpose_uniform4096_kernel` | `reports/memdesc_transpose_uniform4096.ncu-rep` |

The reports can be opened with Nsight Compute 2026.2. Results may vary with the GPU, driver, CUDA, Triton version, and runtime configuration; the archived reports document the results of this experiment.

## Repository Layout

```text
triton-tle-attention-optimization-demo/
├── README.md
├── README_cn.md
├── pyproject.toml
├── requirements-experiment.txt
├── benchmark/
│   ├── bench_rswgmma.py
│   └── bench_memdesc_transpose.py
├── reports/
│   ├── rswgmma_uniform4096.ncu-rep
│   └── memdesc_transpose_uniform4096.ncu-rep
└── src/
    └── hpc_ops_attention/
        ├── __init__.py
        ├── rswgmma_uniform4096.py
        └── memdesc_transpose_uniform4096.py
```

## Quick Start

### Requirements

The project was validated with:

```text
GPU             NVIDIA H100 (Hopper, SM90)
PyTorch         2.7.0
Triton          3.3.0
FlagTree        0.6.0+git367dc579
CUDA Toolkit    12.8
Nsight Compute  2026.2
Python          3.12.x
```

A compatible NVIDIA Driver, a working CUDA toolchain, and the TLE/FlagTree Triton extension are also required.

### Installation

```bash
pip install -e ".[experiment]"
```

Pinned versions are recorded in `requirements-experiment.txt`.

### Run the benchmarks

```bash
python -m pytest -s benchmark/bench_rswgmma.py::test_rswgmma_uniform4096

python -m pytest -s benchmark/bench_memdesc_transpose.py::test_memdesc_transpose_uniform4096
```

## Reproduce NCU Reports

```bash
HPC_OPS_WARMUP=1 HPC_OPS_ITERS=1 ncu --set full --kernel-name-base function \
  -k regex:rswgmma_uniform4096_kernel -c 1 -f \
  -o reports/rswgmma_uniform4096 \
  python -m pytest -s benchmark/bench_rswgmma.py::test_rswgmma_uniform4096

HPC_OPS_WARMUP=1 HPC_OPS_ITERS=1 ncu --set full --kernel-name-base function \
  -k regex:memdesc_transpose_uniform4096_kernel -c 1 -f \
  -o reports/memdesc_transpose_uniform4096 \
  python -m pytest -s benchmark/bench_memdesc_transpose.py::test_memdesc_transpose_uniform4096
```

Set `HPC_OPS_WARMUP=1` and `HPC_OPS_ITERS=1` for a lightweight NCU run; the default values are suitable for standalone timing.
