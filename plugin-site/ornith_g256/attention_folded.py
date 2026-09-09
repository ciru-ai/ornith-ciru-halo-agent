# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru. Optional native query/GQA folding.
# Generated arithmetic retains vLLM/SGLang Apache-2.0 code: Copyright
# contributors to the vLLM project; Copyright2025 vLLM Team;
# Copyright2023-2024 SGLang Team. See installed triton_decode_attention.py.
"""Optional native K/V sharing in eight-query tiles times eight GQA heads.

No installation hook: callers explicitly select this operation.
"""
from functools import lru_cache
import inspect
import linecache

import torch
from vllm.triton_utils import tl, triton
from . import attention_partition as current


@triton.jit
def _reduction_lengths(Starts, Seq, RowSeq,
                       REQUESTS: tl.constexpr, REQUEST_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    req = tl.arange(0, REQUEST_BLOCK)
    first = tl.load(Starts + req, req < REQUESTS, other=0)
    end = tl.load(Starts + req + 1, req < REQUESTS, other=0)
    owns = (req < REQUESTS) & (row >= first) & (row < end)
    owner = tl.max(tl.where(owns, req + 1, 0), 0) - 1
    seq = tl.load(Seq + owner, owner >= 0, other=0)
    # All queries share the request's split boundaries. Causality is applied
    # per row in stage1, rather than changing which partials stage2 visits.
    tl.store(RowSeq + row, seq)


@lru_cache(maxsize=2)
def _kernel(query_tiles=1):
    if query_tiles not in (1, 2):
        raise ValueError('Folded attention supports one or two eight-query tiles')
    native, reduce = current._kernels()
    source = inspect.getsource(native.fn)
    replacements = {
        'def _fwd_grouped_kernel_stage1(': 'def _folded_query_stage1(',
        '    B_Seqlen,\n': '    B_Seqlen,\n    Query_Start,\n',
        '''    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)''':
        '''    cur_kv_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    query_first = tl.load(Query_Start + cur_batch)
    query_end = tl.load(Query_Start + cur_batch + 1)
    query_count = query_end - query_first
    if query_count <= 0:
        return
    offs_m = tl.arange(0, BLOCK_H)
    cur_query = query_first + offs_m // 8
    cur_head = cur_kv_head * 8 + offs_m % 8
    mask_h = offs_m // 8 < query_count''',
        '    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)\n':
        '    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)\n'
        '    causal_len = cur_batch_seq_len - query_count + offs_m // 8 + 1\n',
        'cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]':
        'cur_query[:, None] * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]',
        'mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")':
        'mask_h[:, None] & (offs_n[None, :] < split_kv_end) & (offs_n[None, :] < causal_len[:, None]), qk, float("-inf")',
        'cur_batch * stride_mid_ob\n            + cur_head[:, None] * stride_mid_oh':
        'cur_query[:, None] * stride_mid_ob\n            + cur_head[:, None] * stride_mid_oh',
        'cur_batch * stride_mid_ob\n            + cur_head * stride_mid_oh':
        'cur_query * stride_mid_ob\n            + cur_head * stride_mid_oh',
        '            acc / e_sum[:, None],':
        '            tl.where(causal_len[:, None] > split_kv_start, acc / e_sum[:, None], 0.),',
        '            e_max + tl.log(e_sum),':
        '            tl.where(causal_len > split_kv_start, e_max + tl.log(e_sum), -float("inf")),',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Unsupported folded-query source anchor: ' + old[:80])
        source = source.replace(old, new)
    kernel_name = '_folded_query_stage1'
    if query_tiles == 2:
        # Keep the original <=8-query specialization byte-for-byte. For up to
        # sixteen queries, each request has two otherwise identical 64-row
        # tiles. query_count remains the TOTAL request query count: shrinking
        # it to the tile size would shift the causal positions of tile zero.
        tiled_replacements = {
            '    cur_batch = tl.program_id(0)\n':
            '    cur_batch = tl.program_id(0) // 2\n'
            '    query_tile = tl.program_id(0) % 2\n',
            '    if query_count <= 0:\n':
            '    if query_count <= query_tile * 8:\n',
            '    cur_query = query_first + offs_m // 8\n':
            '    cur_query = query_first + query_tile * 8 + offs_m // 8\n',
            '    mask_h = offs_m // 8 < query_count\n':
            '    mask_h = query_tile * 8 + offs_m // 8 < query_count\n',
            '    causal_len = cur_batch_seq_len - query_count + offs_m // 8 + 1\n':
            '    causal_len = cur_batch_seq_len - query_count + query_tile * 8 + offs_m // 8 + 1\n',
            'def _folded_query_stage1(': 'def _folded_query_tiles_stage1(',
        }
        for old, new in tiled_replacements.items():
            if source.count(old) != 1:
                raise RuntimeError('Unsupported folded query-tile source anchor: ' + old[:80])
            source = source.replace(old, new)
        kernel_name = '_folded_query_tiles_stage1'
    filename = __file__ + ('.generated' if query_tiles == 1 else '.query_tiles.generated')
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(native.fn.__globals__)
    exec(compile(source, filename, 'exec'), namespace)
    return namespace[kernel_name], reduce


def forward(query, key_cache, value_cache, output, block_table, query_start_loc,
            seq_lens, sm_scale, k_scale, v_scale, *, max_query_len=8):
    """Match attention_partition.forward, with native current KV prewritten.

    Caller eligibility: causal target attention, no sinks/alibi/window/output
    scaling, at most8 requests and128 total rows, max_query_len<=16. Query
    starts include empty slots; trailing padded rows receive zero outputs.
    Tensor metadata is trusted as in the ROCm backend; no host reads.
    Scratch remains [query,16,32,257] FP32 plus LSE and per-query reducer extent.
    This is R*526404 bytes and does not copy the block table or KV pool.
    """
    page_size = key_cache.shape[3] if key_cache.ndim == 5 else 0
    if (query.ndim != 3 or query.shape[1:] != (16, 256)
            or page_size not in (1120, 2240) or key_cache.shape[1:] != (2, 32, page_size, 8)
            or value_cache.ndim != 4 or value_cache.shape[1:] != (2, 256, page_size)
            or key_cache.shape[0] != value_cache.shape[0]
            or key_cache.shape[0] == 0 or output.shape != query.shape):
        raise ValueError("Folded attention requires target H16/KV2/D256/page1120 or2240")
    if (not 1 <= max_query_len <= 16 or query.shape[0] > 128
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
    query_tiles = 1 if max_query_len <= 8 else 2
    stage1, reduce = _kernel(query_tiles)
    row_seq = torch.empty((rows,), dtype=torch.int32, device=query.device)
    _reduction_lengths[(rows,)](query_start_loc, seq_lens, row_seq,
        REQUESTS=seq_lens.numel(), REQUEST_BLOCK=triton.next_power_of_2(seq_lens.numel()),
        num_warps=4)
    splits = 32
    logits = torch.empty((rows, 16, splits, 257), dtype=torch.float32, device=query.device)
    lse = torch.empty((rows, 16), dtype=torch.float32, device=query.device)
    stage1[(seq_lens.numel() * query_tiles, 2, splits)](
        query, key_cache, value_cache, sm_scale, block_table, seq_lens, query_start_loc, logits,
        block_table.stride(0), query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(3), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(3), value_cache.stride(1),
        logits.stride(0), logits.stride(1), logits.stride(2), k_scale, v_scale,
        kv_group_num=8, q_head_num=16, BLOCK_DMODEL=256, BLOCK_DPE=0,
        BLOCK_DV=256, BLOCK_N=16, BLOCK_H=64, NUM_KV_SPLITS=splits,
        PAGE_SIZE=page_size, logit_cap=0., Lk=256, Lv=256, IS_MLA=False,
        stride_buf_kds=key_cache.stride(2), stride_buf_kxs=key_cache.stride(4),
        stride_buf_vds=value_cache.stride(2), num_warps=4, num_stages=1,
        waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2)
    reduce(logits, query, output, lse, value_cache.transpose(2, 3), row_seq, splits)
    return output
