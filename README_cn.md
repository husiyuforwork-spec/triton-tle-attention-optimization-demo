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

本 Demo 聚焦 `uniform_4096` 工作负载，用于观察不同数据布局处理方式对 FP8 Attention kernel 的影响。仓库包含源码、独立 benchmark、固定实验依赖和可直接打开的 `.ncu-rep` 报告，便于在技术汇报后进行复核和学习。

## 实现对比

| 实现 | 数据布局处理 | 计算路径 |
|---|---|---|
| RS-WGMMA | 寄存器级数据转置 | WGMMA register-source |
| Transposed-Memdesc | 转置数据描述符 | WGMMA register-source |

两种实现使用相同的 `uniform_4096` 场景，便于对比 Kernel Duration、指令数量、Shared Memory 访问、Bank Conflict、Occupancy 和内存吞吐等 NCU 指标。

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
python benchmark/bench_rswgmma.py \
  --cases uniform_4096 --layout NHD \
  --warmup 50 --iters 300 --repeat 5

python benchmark/bench_memdesc_transpose.py \
  --cases uniform_4096 --layout NHD \
  --warmup 50 --iters 300 --repeat 5
```

## 复现 NCU 报告

```bash
ncu --set full --kernel-name-base function \
  -k regex:rswgmma_uniform4096_kernel -c 1 -f \
  -o reports/rswgmma_uniform4096 \
  python benchmark/bench_rswgmma.py \
  --cases uniform_4096 --layout NHD \
  --no-graph --warmup 1 --iters 1 --repeat 1

ncu --set full --kernel-name-base function \
  -k regex:memdesc_transpose_uniform4096_kernel -c 1 -f \
  -o reports/memdesc_transpose_uniform4096 \
  python benchmark/bench_memdesc_transpose.py \
  --cases uniform_4096 --layout NHD \
  --no-graph --warmup 1 --iters 1 --repeat 1
```

`-c 1`、`--warmup 1` 和 `--iters 1` 用于生成轻量级复现报告；如需稳定的性能统计，可在不使用 NCU 时提高 warmup、iters 和 repeat。
