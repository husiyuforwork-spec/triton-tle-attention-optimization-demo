"""Uniform_4096 RS-WGMMA Attention decode kernel.

Each CTA computes one 512-token fine-grained task and publishes its normalized
partial to its own shared-memory buffer.  Rank 0 remotely reads ranks 1-7 and
performs one online-softmax merge, so DSM scratch remains constant per CTA as
the cluster grows from four to eight ranks.
"""
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from .assign_task import CLUSTER_SIZE, ROWS_Q, TASK_SLOTS, TASK_STRIDE

_TASK_STRIDE_JIT = tl.constexpr(TASK_STRIDE)
_TASK_SLOTS_JIT = tl.constexpr(TASK_SLOTS)
_ROWS_Q_JIT = tl.constexpr(ROWS_Q)
_TMA_STAGES_JIT = tl.constexpr(2)

@triton.jit
def rswgmma_uniform4096_kernel(Q, K_DESC, VT_DESC, BLOCK_IDS, TASK_MAP, QSCALE, KSCALE, VSCALE, OUT, mesh: tl.constexpr, B: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, BLOCK_N: tl.constexpr, Q_STRIDE_B: tl.constexpr, Q_STRIDE_H: tl.constexpr, QS_STRIDE_B: tl.constexpr, QS_STRIDE_H: tl.constexpr, O_STRIDE_B: tl.constexpr, O_STRIDE_H: tl.constexpr):
    cta = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, 'cluster_x')
    task_base = (cta * _TASK_SLOTS_JIT + 1) * _TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task_base + 0)
    batch = tl.load(TASK_MAP + task_base + 1)
    seq_start = tl.load(TASK_MAP + task_base + 3)
    seq_len = tl.load(TASK_MAP + task_base + 4)
    seq_kvcache = tl.load(TASK_MAP + task_base + 5)
    is_causal = tl.load(TASK_MAP + task_base + 8)
    q_smem = tle.gpu.alloc([_ROWS_Q_JIT, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_ROWS_Q_JIT, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc([_TMA_STAGES_JIT, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    v_raw_smem = tle.gpu.alloc([_TMA_STAGES_JIT, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=_TMA_STAGES_JIT, arrive_count=1, expect_bytes=BLOCK_N * D)
    vt_full = tle.gpu.alloc_barriers(num_barriers=_TMA_STAGES_JIT, arrive_count=1, expect_bytes=DV * BLOCK_N)
    if 8 == 2:
        peer_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    elif 8 == 4:
        peer1_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer2_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer3_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer1_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer2_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer3_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    else:
        partial_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        partial_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_q = tl.arange(0, _ROWS_Q_JIT)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    q_rows = tl.broadcast_to(tl.arange(0, _ROWS_Q_JIT)[:, None], (_ROWS_Q_JIT, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (_ROWS_Q_JIT, D))
    p_rows = tl.broadcast_to(tl.arange(0, _ROWS_Q_JIT)[:, None], (_ROWS_Q_JIT, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (_ROWS_Q_JIT, BLOCK_N))
    acc_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, _ROWS_Q_JIT))
    acc_cols = tl.broadcast_to(tl.arange(0, _ROWS_Q_JIT)[None, :], (DV, _ROWS_Q_JIT))
    store_offs_v = offs_v
    q_smem_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    if 8 == 8:
        partial_acc_ptr = tle.gpu.local_ptr(partial_acc_smem, (acc_rows, acc_cols))
        partial_lse_ptr = tle.gpu.local_ptr(partial_lse_smem, (offs_q,))
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE + 0).to(tl.float32)
    vscale = tl.load(VSCALE + 0).to(tl.float32) / 256.0
    hq = hkv * HEADS_PER_GROUP + offs_q
    valid_q = True & (offs_q < HEADS_PER_GROUP) & (hq < H_Q)
    acc = tl.zeros((DV, _ROWS_Q_JIT), tl.float32)
    lse = tl.full((_ROWS_Q_JIT,), -float('inf'), tl.float32)
    q = tl.load(Q + batch * Q_STRIDE_B + hq[:, None] * Q_STRIDE_H + offs_d[None, :], mask=valid_q[:, None], other=0.0)
    tl.store(q_smem_ptr, q)
    qscale = tl.load(QSCALE + batch * QS_STRIDE_B + hq * QS_STRIDE_H, mask=valid_q, other=1.0).to(tl.float32)
    m_i = tl.full((_ROWS_Q_JIT,), -float('inf'), tl.float32)
    l_i = tl.zeros((_ROWS_Q_JIT,), tl.float32)
    copy_iter = 0
    start = 0
    if start < seq_len:
        block_no = seq_start // BLOCK_SIZE
        phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
        tle.gpu.copy(K_DESC, k_raw_smem.slot(0), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(VT_DESC, v_raw_smem.slot(0), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[0])
    while start < seq_len:
        local_n = start + offs_n
        valid_cols = local_n < seq_len
        buf = copy_iter % _TMA_STAGES_JIT
        phase = copy_iter // _TMA_STAGES_JIT & 1
        next_start = start + BLOCK_N
        if next_start < seq_len:
            next_iter = copy_iter + 1
            next_buf = next_iter % _TMA_STAGES_JIT
            aligned_logical = seq_start + next_start
            block_no = aligned_logical // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            tle.gpu.copy(K_DESC, k_raw_smem.slot(next_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[next_buf])
            tle.gpu.copy(VT_DESC, v_raw_smem.slot(next_buf), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[next_buf])
        tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
        k_page = k_raw_smem.slot(buf)
        scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores)
        scores = scores * (inv_sqrt_d * kscale * 1.4426950408889634)
        scores = scores * qscale[None, :]
        causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + 1)
        scores = tl.where(valid_cols[:, None] & causal & valid_q[None, :], scores, -float('inf'))
        m_tile = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        valid_update = m_new != -float('inf')
        safe_m_new = tl.where(valid_update, m_new, 0.0)
        safe_m_i = tl.where(m_i == -float('inf'), safe_m_new, m_i)
        p = tl.exp2(scores - safe_m_new[None, :])
        p = tl.where(valid_update[None, :], p, 0.0)
        alpha = tl.exp2(safe_m_i - safe_m_new)
        alpha = tl.where(valid_update, alpha, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        p_scaled_t = tl.trans((p * 256.0).to(tl.float8e4nv))
        tl.store(p_smem_ptr, p_scaled_t)
        tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
        v_page = v_raw_smem.slot(buf)
        v_page_reg_t = tl.zeros((DV, BLOCK_N), dtype=tl.float8e4nv)
        v_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, DV))
        v_cols = tl.broadcast_to(tl.arange(0, DV)[None, :], (BLOCK_N, DV))
        v_page_reg_t = tl.trans(tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols))))
        pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[None, :] + pv
        m_i = m_new
        l_i = l_new
        start = next_start
        copy_iter += 1
    has_value = l_i > 0.0
    acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float('inf'))
    group_acc = acc
    group_lse = lse
    if 8 == 2:
        peer_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
        peer_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer_lse_remote, (offs_q,)), lse)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float('inf')
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float('inf'), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(peer_lse != -float('inf'), tl.exp2(peer_lse - safe_max), 0.0)
            denom = weight0 + weight1
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            group_acc = (acc * weight0[None, :] + peer_acc * weight1[None, :]) / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float('inf'))
    elif 8 == 4:
        peer1_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        peer2_acc_remote = tle.remote(peer2_acc_smem, 0, scope=mesh)
        peer3_acc_remote = tle.remote(peer3_acc_smem, 0, scope=mesh)
        peer1_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        peer2_lse_remote = tle.remote(peer2_lse_smem, 0, scope=mesh)
        peer3_lse_remote = tle.remote(peer3_lse_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer1_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer1_lse_remote, (offs_q,)), lse)
        elif cluster_rank == 2:
            tl.store(tle.gpu.local_ptr(peer2_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer2_lse_remote, (offs_q,)), lse)
        elif cluster_rank == 3:
            tl.store(tle.gpu.local_ptr(peer3_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer3_lse_remote, (offs_q,)), lse)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
            max_lse = tl.maximum(tl.maximum(lse, lse1), tl.maximum(lse2, lse3))
            valid_group = max_lse != -float('inf')
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float('inf'), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float('inf'), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float('inf'), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float('inf'), tl.exp2(lse3 - safe_max), 0.0)
            denom = weight0 + weight1 + weight2 + weight3
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight1[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight2[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight3[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float('inf'))
    else:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        tle.distributed_barrier(mesh)
        peer1_acc = tle.remote(partial_acc_smem, 1, scope=mesh)
        peer2_acc = tle.remote(partial_acc_smem, 2, scope=mesh)
        peer3_acc = tle.remote(partial_acc_smem, 3, scope=mesh)
        peer4_acc = tle.remote(partial_acc_smem, 4, scope=mesh)
        peer5_acc = tle.remote(partial_acc_smem, 5, scope=mesh)
        peer6_acc = tle.remote(partial_acc_smem, 6, scope=mesh)
        peer7_acc = tle.remote(partial_acc_smem, 7, scope=mesh)
        peer1_lse = tle.remote(partial_lse_smem, 1, scope=mesh)
        peer2_lse = tle.remote(partial_lse_smem, 2, scope=mesh)
        peer3_lse = tle.remote(partial_lse_smem, 3, scope=mesh)
        peer4_lse = tle.remote(partial_lse_smem, 4, scope=mesh)
        peer5_lse = tle.remote(partial_lse_smem, 5, scope=mesh)
        peer6_lse = tle.remote(partial_lse_smem, 6, scope=mesh)
        peer7_lse = tle.remote(partial_lse_smem, 7, scope=mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse, (offs_q,)))
            lse4 = tl.load(tle.gpu.local_ptr(peer4_lse, (offs_q,)))
            lse5 = tl.load(tle.gpu.local_ptr(peer5_lse, (offs_q,)))
            lse6 = tl.load(tle.gpu.local_ptr(peer6_lse, (offs_q,)))
            lse7 = tl.load(tle.gpu.local_ptr(peer7_lse, (offs_q,)))
            max_lse = tl.maximum(lse, lse1)
            max_lse = tl.maximum(max_lse, lse2)
            max_lse = tl.maximum(max_lse, lse3)
            max_lse = tl.maximum(max_lse, lse4)
            max_lse = tl.maximum(max_lse, lse5)
            max_lse = tl.maximum(max_lse, lse6)
            max_lse = tl.maximum(max_lse, lse7)
            valid_group = max_lse != -float('inf')
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float('inf'), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float('inf'), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float('inf'), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float('inf'), tl.exp2(lse3 - safe_max), 0.0)
            weight4 = tl.where(lse4 != -float('inf'), tl.exp2(lse4 - safe_max), 0.0)
            weight5 = tl.where(lse5 != -float('inf'), tl.exp2(lse5 - safe_max), 0.0)
            weight6 = tl.where(lse6 != -float('inf'), tl.exp2(lse6 - safe_max), 0.0)
            weight7 = tl.where(lse7 != -float('inf'), tl.exp2(lse7 - safe_max), 0.0)
            denom = weight0 + weight1 + weight2 + weight3 + weight4 + weight5 + weight6 + weight7
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer1_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight1[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer2_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight2[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer3_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight3[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer4_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight4[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer5_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight5[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer6_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight6[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer7_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight7[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float('inf'))
        tle.distributed_barrier(mesh)
    if cluster_rank == 0:
        output_mask = (offs_q < HEADS_PER_GROUP)[None, :] & (hq < H_Q)[None, :]
        tl.store(OUT + batch * O_STRIDE_B + hq[None, :] * O_STRIDE_H + store_offs_v[:, None], group_acc, mask=output_mask)
__all__ = ['CLUSTER_SIZE', 'rswgmma_uniform4096_kernel']
