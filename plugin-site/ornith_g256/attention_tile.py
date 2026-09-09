# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru. Wrappers around the installed vLLM attention implementation.
"""Small query tiles plus bounded sliding-window context loads for the drafter."""
import hashlib
import inspect
import types
import torch

from vllm.v1.attention.ops import prefix_prefill
from vllm.v1.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode
from .attention_window import build_query_kernel, build_window_kernel
from .attention_partition import forward as partitioned_attention

_compact_prefill = False
_folded_decode = False
_folded_decode_max_queries = 8

original_source = inspect.getsource(prefix_prefill.context_attention_fwd)
replacement = '        BLOCK_M = 16 if 1 <= max_input_len <= 8 else 32'
assert original_source.count('        BLOCK_M = 32') == 1
modified_source = original_source.replace('        BLOCK_M = 32', replacement)
assert modified_source.count('_fwd_kernel[') == 1
modified_source = modified_source.replace(
    '_fwd_kernel[',
    '(_window_kernel if sliding_window > 0 else _query_kernel)[')
helper_namespace = dict(vars(prefix_prefill))
helper_namespace['_window_kernel'] = build_window_kernel(prefix_prefill)
helper_namespace['_query_kernel'] = build_query_kernel(prefix_prefill)
exec(compile(modified_source, __file__ + ':context_attention_fwd', 'exec'), helper_namespace)
context_attention_fwd_tile16 = helper_namespace['context_attention_fwd']
caller_namespace = dict(chunked_prefill_paged_decode.__globals__)
caller_namespace['context_attention_fwd'] = context_attention_fwd_tile16
_tiled_attention = types.FunctionType(
    chunked_prefill_paged_decode.__code__, caller_namespace,
    'chunked_prefill_paged_decode_tile16', chunked_prefill_paged_decode.__defaults__,
    chunked_prefill_paged_decode.__closure__)
_tiled_attention.__kwdefaults__ = chunked_prefill_paged_decode.__kwdefaults__


def chunked_prefill_paged_decode_tile16(
    query, key, value, output, kv_cache_dtype, key_cache, value_cache,
    block_table, query_start_loc, seq_lens, max_seq_len, max_query_len,
    k_scale, v_scale, alibi_slopes=None, sliding_window=None, sm_scale=None,
    output_scale=None, sinks=None, is_block_table_ptr=False, causal=True,
):
    page_size = key_cache.shape[3] if key_cache.ndim == 5 else 0
    target_cache = (page_size in (1120, 2240)
                    and key_cache.shape[1:] == (2, 32, page_size, 8)
                    and value_cache.shape[1:] == (2, 256, page_size))
    # Current K/V has already been written by the ROCm backend. Split only
    # mixed target calls, preserving the existing per-shape attention choices.
    if (_compact_prefill and _folded_decode and target_cache
            and query.shape[1:] == (16, 256)
            and query.dtype == key_cache.dtype == value_cache.dtype == output.dtype == torch.bfloat16
            and kv_cache_dtype == 'auto' and causal and not is_block_table_ptr
            and alibi_slopes is None and sinks is None and output_scale is None
            and (sliding_window is None or sliding_window <= 0)):
        from .attention_mixed import try_forward as mixed_forward
        if mixed_forward(chunked_prefill_paged_decode_tile16,
                query, key, value, output, kv_cache_dtype, key_cache, value_cache,
                block_table, query_start_loc, seq_lens, max_seq_len, max_query_len,
                k_scale, v_scale, alibi_slopes, sliding_window, sm_scale,
                output_scale, sinks, is_block_table_ptr, causal):
            return output
    # Q1/C1 uses the existing partitioned decoder; shape-only eligibility
    # is identical during graph capture and replay as sequence length grows.
    if (_folded_decode and max_query_len == 1
            and query.shape == (1,16,256) and seq_lens.numel() == 1
            and target_cache
            and query.dtype == key_cache.dtype == value_cache.dtype == output.dtype == torch.bfloat16
            and kv_cache_dtype == 'auto' and causal and not is_block_table_ptr
            and alibi_slopes is None and sinks is None and output_scale is None
            and (sliding_window is None or sliding_window <= 0)):
        return partitioned_attention(query,key_cache,value_cache,output,
            block_table,query_start_loc,seq_lens,
            sm_scale if sm_scale is not None else 256**-.5,
            k_scale,v_scale,max_query_len=1)
    # A 16-token verification block is two efficient eight-query/GQA tiles.
    # Route it before generic prefill; the latter wastes the intended decode
    # reuse and should not determine this block size's performance potential.
    if (_folded_decode and 8 < max_query_len <= _folded_decode_max_queries
            and query.shape[0] <= 128 and seq_lens.numel() <= 8
            and query.shape[1:] == (16, 256)
            and target_cache
            and query.dtype == key_cache.dtype == value_cache.dtype == output.dtype == torch.bfloat16
            and kv_cache_dtype == 'auto' and causal and not is_block_table_ptr
            and alibi_slopes is None and sinks is None and output_scale is None
            and (sliding_window is None or sliding_window <= 0)):
        from .attention_folded import forward
        return forward(query, key_cache, value_cache, output, block_table,
                       query_start_loc, seq_lens,
                       sm_scale if sm_scale is not None else 256**-.5,
                       k_scale, v_scale, max_query_len=max_query_len)
    if (_compact_prefill and max_query_len > 8
            and query.shape[1:] == (16, 256)
            and target_cache
            and query.dtype == key_cache.dtype == value_cache.dtype == output.dtype == torch.bfloat16
            and kv_cache_dtype == 'auto' and causal and not is_block_table_ptr
            and alibi_slopes is None and sinks is None and output_scale is None
            and (sliding_window is None or sliding_window <= 0)):
        from .attention_compact import try_forward
        if try_forward(query, key_cache, value_cache, output, block_table,
                       query_start_loc, seq_lens, max_query_len=max_query_len,
                       max_seq_len=max_seq_len,
                       sm_scale=sm_scale if sm_scale is not None else 256**-.5):
            return output
        raise RuntimeError('Enabled compact prefill rejected target metadata: ' + str({
            name: (tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
            for name, t in [('query', query), ('output', output), ('table', block_table),
                            ('starts', query_start_loc), ('seq_lens', seq_lens)]
        }) + f'; max_query_len={max_query_len}, max_seq_len={max_seq_len}')
    # Parallel KV partitions hide the long-latency accesses of dispersed hybrid
    # cache pages. Larger query batches reuse K/V better in the tiled prefill
    # kernel; Q106 was slower with virtual decode, so keep the measured cutoff.
    if (1 < max_query_len <= 8 and max_seq_len >= 8192
            and query.shape[0] <= 64 and seq_lens.numel() <= 8
            and query.shape[1:] == (16, 256)
            and target_cache
            and query.dtype == key_cache.dtype == value_cache.dtype == output.dtype == torch.bfloat16
            and kv_cache_dtype == 'auto' and causal and not is_block_table_ptr
            and alibi_slopes is None and sinks is None and output_scale is None
            and (sliding_window is None or sliding_window <= 0)):
        if _folded_decode:
            from .attention_folded import forward as decode
        else:
            decode = partitioned_attention
        return decode(
            query, key_cache, value_cache, output, block_table, query_start_loc,
            seq_lens, sm_scale if sm_scale is not None else 256**-.5,
            k_scale, v_scale, max_query_len=max_query_len)
    return _tiled_attention(
        query, key, value, output, kv_cache_dtype, key_cache, value_cache,
        block_table, query_start_loc, seq_lens, max_seq_len, max_query_len,
        k_scale, v_scale, alibi_slopes, sliding_window, sm_scale, output_scale,
        sinks, is_block_table_ptr, causal)


provenance = dict(source_file=prefix_prefill.__file__,
    original_helper_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
    modified_helper_sha256=hashlib.sha256(modified_source.encode()).hexdigest(),
    exact_change=replacement.strip(),
    window_change='Skip and mask context K/V outside the earliest query-row window',
    query_change='Return before context work for query tiles with no output rows',
    decode_change='Target page1120/2240, context>=8192, maxQ2..8: native-layout32-way KV partitioning; optional maxQ16 folding',
    unchanged='Full-attention arithmetic, BLOCK_N32, cache tile32, four warps, one stage')


def install(*, compact_prefill=False, folded_decode=False, folded_decode_max_queries=8):
    """Select this helper for target and draft ROCm attention in this worker."""
    global _compact_prefill, _folded_decode, _folded_decode_max_queries
    if folded_decode_max_queries not in (8, 16):
        raise ValueError('Folded verification supports eight or sixteen query positions')
    _compact_prefill = compact_prefill
    _folded_decode = folded_decode
    _folded_decode_max_queries = folded_decode_max_queries
    from vllm.v1.attention.backends import rocm_attn
    current = rocm_attn.chunked_prefill_paged_decode
    if current not in (chunked_prefill_paged_decode, chunked_prefill_paged_decode_tile16):
        raise RuntimeError('Another extension replaced the ROCm prefill helper')
    rocm_attn.chunked_prefill_paged_decode = chunked_prefill_paged_decode_tile16
