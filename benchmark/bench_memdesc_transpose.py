"""Transposed-memdesc RS-WGMMA benchmark with a fixed execution path."""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

PROJECT_ROOT = Path(os.environ.get("HPC_OPS_ROOT", Path(__file__).resolve().parents[1]))
ATTENTION_BENCH_DIR = PROJECT_ROOT / "benchmark" / "attention_decode"
sys.path.insert(0, str(ATTENTION_BENCH_DIR))

from bench_attention_decode_fp8_tle_v35_task_aware_fused import (
    BLOCK_SIZE, CASES, DEFAULT_HEAD_DIM, DEFAULT_KV_HEADS, DEFAULT_NUM_SEQ_Q,
    DEFAULT_Q_HEADS, TILE_N, _make_inputs_compat, _run_cuda_kernel_compat,
    bench_us, bench_us_mean, make_task_map,
)
from bench_attention_decode_fp8_tle_v43_cluster8_owner_local_fused import (
    V43_CLUSTER_MESH, prepare_v43_workspace,
)

KERNEL_DIR = Path(__file__).resolve().parents[1] / "src" / "hpc_ops_attention"
sys.path.insert(0, str(KERNEL_DIR))
from memdesc_transpose_uniform4096 import memdesc_transpose_uniform4096_kernel  # noqa: E402


def descriptors(inputs):
    block_shape = [1, 1, TILE_N, inputs.q.shape[2]]
    return (
        TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=block_shape),
        TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=block_shape),
    )


def run(inputs, workspace, k_desc, v_desc):
    ws = workspace.base
    memdesc_transpose_uniform4096_kernel[(workspace.num_clusters,)](
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
    return ws.out.reshape(inputs.num_batch, inputs.q.shape[1], inputs.v_cache.shape[3])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    p.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    p.add_argument("--num-head-kv", type=int, default=DEFAULT_KV_HEADS)
    p.add_argument("--num-head-q", type=int, default=DEFAULT_Q_HEADS)
    p.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--min-process-len", type=int, default=512)
    p.add_argument("--check", action="store_true")
    p.add_argument("--no-graph", dest="graph", action="store_false")
    p.set_defaults(graph=True)
    args = p.parse_args()
    measure = bench_us if args.graph else lambda fn, w, i, _g: bench_us_mean(fn, w, i)
    for case in args.cases:
        inputs = _make_inputs_compat(args, case)
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=inputs.q.device))
        if args.layout == "HND":
            inputs.k_cache = inputs.k_cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
            inputs.v_cache = inputs.v_cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        k_desc, v_desc = descriptors(inputs)
        workspace = prepare_v43_workspace(inputs, chunk_tokens=512)
        task_map = make_task_map(inputs.kv_lens, args.num_head_kv, DEFAULT_NUM_SEQ_Q, args.min_process_len)
        actual = lambda: run(inputs, workspace, k_desc, v_desc)
        reference = lambda: _run_cuda_kernel_compat(inputs, task_map)
        if args.check:
            expected = reference().detach().clone(); result = actual(); torch.cuda.synchronize()
            assert torch.allclose(result, expected, atol=0.2, rtol=0.2)
        samples = [measure(actual, args.warmup, args.iters, args.graph) for _ in range(args.repeat)]
        print(f"{case} | Transposed-Memdesc | {statistics.median(samples):.3f} us")


if __name__ == "__main__":
    main()
