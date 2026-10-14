"""Pytest benchmark for the uniform_4096 RS-WGMMA Attention kernel."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

PROJECT_ROOT = Path(os.environ.get("HPC_OPS_ROOT", Path(__file__).resolve().parents[1]))
ATTENTION_BENCH_DIR = PROJECT_ROOT / "benchmark" / "attention_decode"
sys.path.insert(0, str(ATTENTION_BENCH_DIR))

from bench_attention_decode_fp8_tle_v35_task_aware_fused import (  # noqa: E402
    BLOCK_SIZE,
    DEFAULT_HEAD_DIM,
    DEFAULT_KV_HEADS,
    DEFAULT_Q_HEADS,
    TILE_N,
    _make_inputs_compat,
)
from bench_attention_decode_fp8_tle_v43_cluster8_owner_local_fused import (  # noqa: E402
    V43_CLUSTER_MESH,
    prepare_v43_workspace,
)

KERNEL_DIR = Path(__file__).resolve().parents[1] / "src" / "hpc_ops_attention"
sys.path.insert(0, str(KERNEL_DIR))
from rswgmma_uniform4096 import rswgmma_uniform4096_kernel  # noqa: E402


CASE_NAME = "uniform_4096"
WARMUP = int(os.environ.get("HPC_OPS_WARMUP", "50"))
ITERATIONS = int(os.environ.get("HPC_OPS_ITERS", "300"))


def _descriptors(inputs):
    block_shape = [1, 1, TILE_N, inputs.q.shape[2]]
    return (
        TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=block_shape),
        TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=block_shape),
    )


def _build_workload():
    args = SimpleNamespace(
        num_head_kv=DEFAULT_KV_HEADS,
        num_head_q=DEFAULT_Q_HEADS,
        head_dim=DEFAULT_HEAD_DIM,
    )
    inputs = _make_inputs_compat(args, CASE_NAME)
    k_desc, v_desc = _descriptors(inputs)
    workspace = prepare_v43_workspace(inputs, chunk_tokens=512)

    def invoke():
        ws = workspace.base
        rswgmma_uniform4096_kernel[(workspace.num_clusters,)](
            ws.q_4d, k_desc, v_desc, inputs.block_ids, ws.task_map, ws.q_scale_3d,
            inputs.k_scale, inputs.v_scale, ws.out, mesh=V43_CLUSTER_MESH,
            B=inputs.num_batch, H_Q=inputs.q.shape[1], HEADS_PER_GROUP=ws.heads_per_group,
            D=inputs.q.shape[2], DV=inputs.v_cache.shape[3], BLOCK_SIZE=BLOCK_SIZE,
            MAX_BLOCKS=inputs.block_ids.shape[1], BLOCK_N=TILE_N,
            Q_STRIDE_B=ws.q_4d.stride(0), Q_STRIDE_H=ws.q_4d.stride(2),
            QS_STRIDE_B=ws.q_scale_3d.stride(0), QS_STRIDE_H=ws.q_scale_3d.stride(2),
            O_STRIDE_B=ws.out.stride(0), O_STRIDE_H=ws.out.stride(2),
            num_ctas=1, num_warps=4, num_stages=3, launch_pdl=False,
        )

    return invoke


@pytest.fixture(scope="module")
def workload():
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    return _build_workload()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rswgmma_uniform4096(workload):
    for _ in range(WARMUP):
        workload()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERATIONS):
        workload()
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0 / ITERATIONS
    print(f"uniform_4096 | RS-WGMMA | {elapsed_us:.3f} us")
