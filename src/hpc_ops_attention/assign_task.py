"""Fixed input and task assignment for the uniform_4096 demo workload."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton.experimental.tle.language as tle


BLOCK_SIZE = 64
TILE_N = 64
CHUNK_TOKENS = 512
NUM_BATCH = 64
KV_LENGTH = 4096
NUM_QUERY_TOKENS = 1
NUM_KV_HEADS = 1
NUM_QUERY_HEADS = 8
HEAD_DIM = 128

TASK_STRIDE = 12
TASK_SLOTS = 2
ROWS_Q = 8
CLUSTER_SIZE = 8
GROUP_MODE = 1

CLUSTER_MESH = tle.device_mesh({"block_cluster": [("cluster_x", CLUSTER_SIZE)]})


@dataclass(frozen=True)
class Uniform4096Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def num_batch(self) -> int:
        return NUM_BATCH


@dataclass(frozen=True)
class Uniform4096Workspace:
    q_4d: torch.Tensor
    q_scale_3d: torch.Tensor
    task_workspace: torch.Tensor
    task_map: torch.Tensor
    out: torch.Tensor
    heads_per_group: int
    num_clusters: int


def _make_block_ids(device: torch.device) -> torch.Tensor:
    blocks_per_sequence = KV_LENGTH // BLOCK_SIZE
    total_blocks = NUM_BATCH * blocks_per_sequence
    storage_blocks = int(total_blocks * 1.2) + NUM_BATCH + 8
    packed = torch.randperm(storage_blocks, device=device)[:total_blocks].to(torch.int32)
    return packed.reshape(NUM_BATCH, blocks_per_sequence)


def make_uniform4096_inputs(device: str | torch.device = "cuda") -> Uniform4096Inputs:
    """Create the single input shape covered by this repository."""
    device = torch.device(device)
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    q = torch.randn(
        (NUM_BATCH * NUM_QUERY_TOKENS, NUM_QUERY_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ) / math.sqrt(HEAD_DIM)
    q_scale = q.float().abs().amax(dim=-1).clamp_min(1e-6)
    q = (q / q_scale[..., None]).to(torch.float8_e4m3fn)

    block_ids = _make_block_ids(device)
    storage_blocks = int(block_ids.max().item()) + 1
    k_cache = torch.randn(
        (storage_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    k_cache = (k_cache / math.sqrt(HEAD_DIM)).to(torch.float8_e4m3fn)
    v_cache = v_cache.to(torch.float8_e4m3fn)

    return Uniform4096Inputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_ids=block_ids,
        kv_lens=torch.full((NUM_BATCH,), KV_LENGTH, dtype=torch.int32, device=device),
        q_scale=q_scale,
        k_scale=torch.rand((1,), dtype=torch.float32, device=device).clamp_min(1e-6),
        v_scale=torch.rand((1,), dtype=torch.float32, device=device).clamp_min(1e-6),
    )


def _make_task(batch: int, cluster_rank: int) -> list[int]:
    seq_start = cluster_rank * CHUNK_TOKENS
    is_last = cluster_rank == CLUSTER_SIZE - 1
    seq_kvcache = CHUNK_TOKENS - NUM_QUERY_TOKENS if is_last else CHUNK_TOKENS
    return [
        0,
        batch,
        cluster_rank,
        seq_start,
        CHUNK_TOKENS,
        seq_kvcache,
        CHUNK_TOKENS // TILE_N,
        seq_kvcache // TILE_N,
        int(is_last),
        GROUP_MODE,
        0,
        1,
    ]


def assign_uniform4096_tasks(inputs: Uniform4096Inputs) -> Uniform4096Workspace:
    """Build one eight-CTA cluster for each sequence in the fixed workload."""
    if tuple(inputs.kv_lens.shape) != (NUM_BATCH,) or not bool(
        torch.all(inputs.kv_lens == KV_LENGTH).item()
    ):
        raise ValueError("assign_uniform4096_tasks requires 64 KV sequences of length 4096")

    num_clusters = NUM_BATCH * NUM_KV_HEADS
    physical_ctas = num_clusters * CLUSTER_SIZE
    task_region_ints = (TASK_SLOTS * physical_ctas + 1) * TASK_STRIDE
    chunk_count_ints = NUM_BATCH * NUM_KV_HEADS
    padded_chunk_count_ints = (
        (chunk_count_ints + TASK_STRIDE - 1) // TASK_STRIDE
    ) * TASK_STRIDE
    workspace_ints = task_region_ints + padded_chunk_count_ints

    task_map_cpu = torch.full((workspace_ints,), -1, dtype=torch.int32)
    task_map_cpu[0] = CHUNK_TOKENS // TILE_N + 1
    task_map_cpu[1] = physical_ctas
    task_map_cpu[2] = NUM_KV_HEADS
    task_map_cpu[3] = NUM_BATCH
    task_map_cpu[4] = workspace_ints * 4

    for batch in range(NUM_BATCH):
        for rank in range(CLUSTER_SIZE):
            cta = batch * CLUSTER_SIZE + rank
            task_base = (cta * TASK_SLOTS + 1) * TASK_STRIDE
            task_map_cpu[task_base : task_base + TASK_STRIDE] = torch.tensor(
                _make_task(batch, rank), dtype=torch.int32
            )
            sentinel_base = (cta * TASK_SLOTS + 2) * TASK_STRIDE
            task_map_cpu[sentinel_base : sentinel_base + 2] = -1

    task_map_cpu[task_region_ints : task_region_ints + chunk_count_ints] = 1
    task_workspace = task_map_cpu.to(device=inputs.q.device).view(torch.int8)

    return Uniform4096Workspace(
        q_4d=inputs.q.reshape(NUM_BATCH, NUM_QUERY_TOKENS, NUM_QUERY_HEADS, HEAD_DIM),
        q_scale_3d=inputs.q_scale.reshape(NUM_BATCH, NUM_QUERY_TOKENS, NUM_QUERY_HEADS),
        task_workspace=task_workspace,
        task_map=task_workspace.view(torch.int32),
        out=torch.empty(
            (NUM_BATCH, NUM_QUERY_TOKENS, NUM_QUERY_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=inputs.q.device,
        ),
        heads_per_group=NUM_QUERY_HEADS // NUM_KV_HEADS,
        num_clusters=num_clusters,
    )


def make_uniform4096_workload() -> tuple[Uniform4096Inputs, Uniform4096Workspace]:
    inputs = make_uniform4096_inputs()
    return inputs, assign_uniform4096_tasks(inputs)


__all__ = [
    "BLOCK_SIZE",
    "CLUSTER_MESH",
    "CLUSTER_SIZE",
    "HEAD_DIM",
    "NUM_KV_HEADS",
    "NUM_QUERY_HEADS",
    "ROWS_Q",
    "TASK_SLOTS",
    "TASK_STRIDE",
    "TILE_N",
    "Uniform4096Inputs",
    "Uniform4096Workspace",
    "assign_uniform4096_tasks",
    "make_uniform4096_inputs",
    "make_uniform4096_workload",
]
