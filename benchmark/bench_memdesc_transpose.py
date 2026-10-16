"""Pytest benchmark for the uniform_4096 Transposed-Memdesc Attention kernel."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from hpc_ops_attention.assign_task import (  # noqa: E402
    BLOCK_SIZE,
    CLUSTER_MESH,
    HEAD_DIM,
    NUM_QUERY_HEADS,
    TILE_N,
    make_uniform4096_workload,
)
from hpc_ops_attention.memdesc_transpose_uniform4096 import (  # noqa: E402
    memdesc_transpose_uniform4096_kernel,
)


WARMUP = int(os.environ.get("HPC_OPS_WARMUP", "50"))
ITERATIONS = int(os.environ.get("HPC_OPS_ITERS", "300"))


def _descriptors(inputs):
    block_shape = [1, 1, TILE_N, inputs.q.shape[2]]
    return (
        TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=block_shape),
        TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=block_shape),
    )


def _build_workload():
    inputs, workspace = make_uniform4096_workload()
    k_desc, v_desc = _descriptors(inputs)

    def invoke():
        memdesc_transpose_uniform4096_kernel[(workspace.num_clusters,)](
            workspace.q_4d, k_desc, v_desc, inputs.block_ids, workspace.task_map,
            workspace.q_scale_3d, inputs.k_scale, inputs.v_scale, workspace.out,
            mesh=CLUSTER_MESH, B=inputs.num_batch, H_Q=NUM_QUERY_HEADS,
            HEADS_PER_GROUP=workspace.heads_per_group, D=HEAD_DIM, DV=HEAD_DIM,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_BLOCKS=inputs.block_ids.shape[1], BLOCK_N=TILE_N,
            Q_STRIDE_B=workspace.q_4d.stride(0), Q_STRIDE_H=workspace.q_4d.stride(2),
            QS_STRIDE_B=workspace.q_scale_3d.stride(0),
            QS_STRIDE_H=workspace.q_scale_3d.stride(2),
            O_STRIDE_B=workspace.out.stride(0), O_STRIDE_H=workspace.out.stride(2),
            num_ctas=1, num_warps=4, num_stages=3, launch_pdl=False,
        )

    return invoke


@pytest.fixture(scope="module")
def workload():
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    return _build_workload()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_memdesc_transpose_uniform4096(workload):
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
    print(f"uniform_4096 | Transposed-Memdesc | {elapsed_us:.3f} us")
