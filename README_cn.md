<div align="center">
  <h1>Triton-TLE Attention 优化实践 Demo</h1>
  <p>
    <img alt="GPU" src="https://img.shields.io/badge/GPU-NVIDIA%20H100%20%7C%20SM90-76B900">
    <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.8%2B-76B900">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-3776AB">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
  </p>
</div>

**English version:** [README.md](README.md)

本项目是技术汇报 **《Triton-TLE 优化新型 Attention kernel 算子实战》** 的配套示例，面向技术交流、现场演示和实验复现，展示 Hopper GPU 上 Attention kernel 的两种数据布局优化实现，并提供对应的 Nsight Compute 报告。

<p align="center">
  <a href="#项目简介">项目简介</a> |
  <a href="#实现对比">实现对比</a> |
  <a href="#性能分析材料">性能分析材料</a> |
  <a href="#快速开始">快速开始</a> |
  <a href="#复现-ncu-报告">复现 NCU 报告</a>
</p>

## 项目简介

本 Demo 用于对比优化前的 RS-WGMMA 实现与优化后的 Transposed-Memdesc 实现：RS-WGMMA 将 V tile 加载至寄存器，执行显式寄存器级转置后使用 register-source WGMMA；Transposed-Memdesc 则通过 memory descriptor 表达转置后的数据布局，减少转置过程中的 shared-memory 数据搬运，再执行相同的 WGMMA 计算。仓库包含源码、独立 benchmark、固定实验依赖和可直接打开的 `.ncu-rep` 报告，便于在技术汇报后进行复核和学习。

## 实现对比

| 实现 | 数据布局处理 | 计算路径 |
|---|---|---|
| RS-WGMMA | 寄存器级数据转置 | WGMMA register-source |
| Transposed-Memdesc | 转置数据描述符 | WGMMA register-source |

两种实现使用相同的 `uniform_4096` 场景，便于对比 Kernel Duration、指令数量、Shared Memory 访问、Bank Conflict、Occupancy 和内存吞吐等 NCU 指标。

## 工作负载配置

本 Demo 使用固定的 `uniform_4096` KV cache 配置：

| 参数 | 配置 |
|---|---:|
| Batch size | 64 条序列 |
| KV cache 长度 | 每条序列 4096 个 token |
| 每次 decode 的 query token 数 | 1 |
| Query heads | 8 |
| KV heads | 1 |
| Head dimension | 128 |

## 性能分析材料

| Benchmark | Kernel | NCU 报告 |
|---|---|---|
| `benchmark/bench_rswgmma.py` | `rswgmma_uniform4096_kernel` | `reports/rswgmma_uniform4096.ncu-rep` |
| `benchmark/bench_memdesc_transpose.py` | `memdesc_transpose_uniform4096_kernel` | `reports/memdesc_transpose_uniform4096.ncu-rep` |

报告可使用 Nsight Compute 2026.2 打开。实际结果会受到 GPU、驱动、CUDA、Triton 版本和运行配置影响；仓库中的报告用于记录本次实验结果。

## 目录结构

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
        ├── assign_task.py
        ├── __init__.py
        ├── rswgmma_uniform4096.py
        └── memdesc_transpose_uniform4096.py
```

## 快速开始

### 环境要求

本项目已在以下环境中验证：

```text
GPU             NVIDIA H100 (Hopper, SM90)
PyTorch         2.7.0
Triton          3.3.0
FlagTree        0.6.0+git367dc579
CUDA Toolkit    12.8
Nsight Compute  2026.2
Python          3.12.x
```

同时需要匹配的 NVIDIA Driver、可用的 CUDA 工具链，以及已安装的 TLE/FlagTree Triton 扩展。

### 安装

```bash
pip install -e ".[experiment]"
```

固定版本记录在 `requirements-experiment.txt` 中。

### 运行 benchmark

```bash
python -m pytest -s benchmark/bench_rswgmma.py::test_rswgmma_uniform4096

python -m pytest -s benchmark/bench_memdesc_transpose.py::test_memdesc_transpose_uniform4096
```

## 复现 NCU 报告

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

进行轻量级 NCU 采样时设置 `HPC_OPS_WARMUP=1` 和 `HPC_OPS_ITERS=1`；独立计时时使用默认值即可。
