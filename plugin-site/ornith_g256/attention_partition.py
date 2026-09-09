# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru.
# The generated kernels retain the installed vLLM/SGLang Apache-2.0 code:
# Copyright contributors to the vLLM project; Copyright 2025 vLLM Team;
# Copyright 2023-2024 SGLang Team. See vLLM triton_decode_attention.py.
"""Native-page partitioned attention for target H16/KV2/D256/page1120 or2240.

The backend must write current K/V before calling ``forward``. Each query row
becomes a causal decode request; existing grouped split/reduction arithmetic is
retained. This module has no installation hook and never converts the KV pool.
"""

from functools import lru_cache
import inspect
import linecache

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _query_metadata(
    Starts, Seq, Table, RowSeq, RowTable,
    TABLE_STRIDE: tl.constexpr, COLS: tl.constexpr,
    REQUESTS: tl.constexpr, REQUEST_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.arange(0, REQUEST_BLOCK)
    start = tl.load(Starts + req, req < REQUESTS, other=0)
    end = tl.load(Starts + req + 1, req < REQUESTS, other=0)
    owns = (req < REQUESTS) & (start <= row) & (row < end)
    owner = tl.max(tl.where(owns, req + 1, 0), 0) - 1
    active = owner >= 0
    first = tl.sum(tl.where(owns, start, 0), 0)
    count = tl.sum(tl.where(owns, end - start, 0), 0)
    seq = tl.load(Seq + owner, active, other=0)
    causal_len = tl.where(active, seq - count + row - first + 1, 0)
    tl.store(RowSeq + row, causal_len)
    cols = tl.arange(0, COL_BLOCK)
    page = tl.load(Table + owner * TABLE_STRIDE + cols,
                   active & (cols < COLS) & (cols * PAGE_SIZE < causal_len), other=0)
    tl.store(RowTable + row * COLS + cols, page, cols < COLS)


def _compile(source, namespace, suffix):
    filename = __file__ + suffix
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)


@lru_cache(maxsize=1)
def _kernels():
    from vllm.v1.attention.ops import triton_decode_attention as module

    namespace = dict(vars(module))
    source = inspect.getsource(module._fwd_grouped_kernel_stage1.fn)
    replacements = {
        "    IS_MLA: tl.constexpr = False,":
        "    stride_buf_kds: tl.constexpr,\n    stride_buf_kxs: tl.constexpr,\n"
        "    stride_buf_vds: tl.constexpr,\n    IS_MLA: tl.constexpr = False,",
        "base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]":
        "base_offs_k = cur_kv_head * stride_buf_kh + (offs_d[:, None] // 8) * stride_buf_kds + (offs_d[:, None] % 8) * stride_buf_kxs",
        "base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]":
        "base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :] * stride_buf_vds",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError("Unsupported installed partitioned decoder addressing")
        source = source.replace(old, new)
    _compile(source, namespace, ".stage1.generated")

    # A graph-padded row owns no request. Stage1 then reads no KV or scratch;
    # bypass the reducer's undefined 0/0 result without changing active math.
    source = inspect.getsource(module._fwd_kernel_stage2.fn)
    old = "    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)\n"
    if source.count(old) != 1:
        raise RuntimeError("Unsupported installed partitioned decoder reduction")
    source = source.replace(old, old +
        "    if cur_batch_seq_len <= 0:\n"
        "        pad_d = tl.arange(0, BLOCK_DV)\n"
        "        tl.store(o + cur_batch * stride_obs + cur_head * stride_oh + pad_d, 0., pad_d < Lv)\n"
        "        tl.store(lse + cur_batch * stride_lse_bs + cur_head, -float('inf'))\n"
        "        return\n")
    _compile(source, namespace, ".stage2.generated")
    _compile(inspect.getsource(module._decode_softmax_reducev_fwd), namespace,
             ".reduce.generated")
    return namespace["_fwd_grouped_kernel_stage1"], namespace["_decode_softmax_reducev_fwd"]


def _metadata(block_table, query_start_loc, seq_lens, rows, page_size=1120):
    """Allocate and fill per-query metadata without a device-to-host read."""
    row_table = torch.empty((rows, block_table.shape[1]), dtype=torch.int32,
                            device=block_table.device)
    row_seq = torch.empty((rows,), dtype=torch.int32, device=block_table.device)
    if rows:
        _query_metadata[(rows,)](
            query_start_loc, seq_lens, block_table, row_seq, row_table,
            TABLE_STRIDE=block_table.stride(0), COLS=block_table.shape[1],
            REQUESTS=seq_lens.numel(), REQUEST_BLOCK=triton.next_power_of_2(seq_lens.numel()),
            COL_BLOCK=triton.next_power_of_2(block_table.shape[1]), PAGE_SIZE=page_size,
            num_warps=4)
    return row_table, row_seq


def forward(query, key_cache, value_cache, output, block_table, query_start_loc,
            seq_lens, sm_scale, k_scale, v_scale, *, max_query_len=8):
    """Write and return ``output``; allocate scratch only for this call.

    Caller eligibility: BF16 causal target attention, no sinks/alibi/window/output
    scaling, at most 8 requests and max_query_len <= 8, at most 64 query rows
    including padding. ``query_start_loc`` has
    len(seq_lens)+1 entries and includes zero-query slots; trailing query padding
    beyond its final entry is allowed and gets zero output. Metadata and cached
    page IDs must be valid, as in the ROCm backend. No host synchronization occurs.

    Scratch bytes for R query rows and B table columns: R*(526404 + 4*B),
    including float32 split outputs/LSE and int32 causal lengths/block tables.
    """
    page_size = key_cache.shape[3] if key_cache.ndim == 5 else 0
    if (query.ndim != 3 or query.shape[1:] != (16, 256)
            or page_size not in (1120, 2240) or key_cache.shape[1:] != (2, 32, page_size, 8)
            or value_cache.ndim != 4 or value_cache.shape[1:] != (2, 256, page_size)
            or key_cache.shape[0] != value_cache.shape[0]
            or key_cache.shape[0] == 0 or output.shape != query.shape):
        raise ValueError("Partitioned attention requires target H16/KV2/D256/page1120 or2240")
    if (not 1 <= max_query_len <= 8 or query.shape[0] > 64
            or not 1 <= seq_lens.numel() <= 8
            or block_table.ndim != 2 or block_table.shape[0] < seq_lens.numel()
            or block_table.shape[1] < 1 or query_start_loc.numel() != seq_lens.numel() + 1
            or not seq_lens.is_contiguous() or not query_start_loc.is_contiguous()
            or block_table.stride(1) != 1 or query.stride(2) != 1 or output.stride(2) != 1):
        raise ValueError("Unsupported query metadata or noncontiguous inner dimensions")
    if (not query.is_cuda or any(t.device != query.device for t in
            (key_cache, value_cache, output, block_table, query_start_loc, seq_lens, k_scale, v_scale))
            or any(t.dtype != torch.bfloat16 for t in (query, key_cache, value_cache, output))
            or any(t.dtype not in (torch.int32, torch.int64) for t in
                   (block_table, query_start_loc, seq_lens))):
        raise ValueError("Partitioned attention requires BF16 and integer metadata on one GPU")
    rows = query.shape[0]
    if not rows:
        return output
    stage1, reduce = _kernels()
    table, seq = _metadata(block_table, query_start_loc, seq_lens, rows, page_size)
    splits = 32
    logits = torch.empty((rows, 16, splits, 257), dtype=torch.float32, device=query.device)
    lse = torch.empty((rows, 16), dtype=torch.float32, device=query.device)
    stage1[(rows, 2, splits)](
        query, key_cache, value_cache, sm_scale, table, seq, logits,
        table.stride(0), query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(3), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(3), value_cache.stride(1),
        logits.stride(0), logits.stride(1), logits.stride(2), k_scale, v_scale,
        kv_group_num=8, q_head_num=16, BLOCK_DMODEL=256, BLOCK_DPE=0,
        BLOCK_DV=256, BLOCK_N=16, BLOCK_H=16, NUM_KV_SPLITS=splits,
        PAGE_SIZE=page_size, logit_cap=0., Lk=256, Lv=256, IS_MLA=False,
        stride_buf_kds=key_cache.stride(2), stride_buf_kxs=key_cache.stride(4),
        stride_buf_vds=value_cache.stride(2), num_warps=4, num_stages=1,
        waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2)
    # The reduction helper only reads v_buffer.shape[-1], never cache data.
    reduce(logits, query, output, lse, value_cache.transpose(2, 3), seq, splits)
    return output
