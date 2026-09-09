# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru.
"""Shared dense KV scratch for target prefill with installed AMD Triton flash.

Configure once before profiling/capture. Calls must be serialized across target
layers, as for the existing native arena. The backend retains ownership of KV
writes. This module neither installs a hook nor changes decode dispatch.
"""
import ast
import inspect
import os

os.environ['FLASH_ATTENTION_TRITON_AMD_AUTOTUNE'] = '0'

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger('vllm.ornith_g256.attention_compact')
_arena = None
_prefill = None


@triton.jit
def _kv_boundaries(Starts, Seq, CuK, REQUESTS: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.arange(0, BLOCK)
    start = tl.load(Starts + req, req < REQUESTS, other=0)
    end = tl.load(Starts + req + 1, req < REQUESTS, other=0)
    length = tl.load(Seq + req, req < REQUESTS, other=0)
    length = tl.where(end > start, length, 0)
    boundaries = tl.cumsum(length, 0)
    tl.store(CuK, 0)
    tl.store(CuK + req + 1, boundaries, req < REQUESTS)


@triton.jit
def _gather(K, V, Table, CuK, DenseK, DenseV,
            KBS: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
            KT: tl.constexpr, KX: tl.constexpr,
            VBS: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr, VT: tl.constexpr,
            TABLE_STRIDE: tl.constexpr, TOKENS: tl.constexpr, PAGE_SIZE: tl.constexpr):
    token = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    head = tl.program_id(1)
    req = tl.program_id(2)
    first = tl.load(CuK + req)
    length = tl.load(CuK + req + 1) - first
    if tl.program_id(0) * TOKENS >= length:
        return
    dim = tl.arange(0, 256)
    valid = token < length
    block = tl.load(Table + req * TABLE_STRIDE + token // PAGE_SIZE,
                    valid, other=0).to(tl.int64)
    within = token % PAGE_SIZE
    k_offset = (block[:, None] * KBS + head * KH + (dim[None, :] // 8) * KD
                + within[:, None] * KT + (dim[None, :] % 8) * KX)
    v_offset = (block[:, None] * VBS + head * VH + dim[None, :] * VD
                + within[:, None] * VT)
    key = tl.load(K + k_offset, valid[:, None], other=0)
    value = tl.load(V + v_offset, valid[:, None], other=0)
    offset = ((first + token[:, None]) * 2 + head) * 256 + dim[None, :]
    tl.store(DenseK + offset, key, valid[:, None])
    tl.store(DenseV + offset, value, valid[:, None])


def _graph_safe_prefill():
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import fwd_prefill as module
    if module.AUTOTUNE != 'off':
        raise RuntimeError('Set FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0 before importing AMD flash attention')
    tree = ast.parse(inspect.getsource(module.attention_forward_prefill_triton_impl))

    class RemoveDeviceBoundaryAssertions(ast.NodeTransformer):
        removed = 0

        def visit_Assert(self, node):
            # vLLM owns/validates the GPU metadata. The four wrapper checks
            # read cu[0]/cu[-1] on the host, preventing graph capture. Capacity
            # tails are also legal here: cu[-1] need not equal tensor capacity.
            if any(isinstance(child, ast.Subscript)
                   and isinstance(child.value, ast.Name)
                   and child.value.id in ('cu_seqlens_q', 'cu_seqlens_k')
                   for child in ast.walk(node.test)):
                self.removed += 1
                return None
            return node

    transform = RemoveDeviceBoundaryAssertions()
    tree = transform.visit(tree)
    if transform.removed != 4:
        raise RuntimeError('Unsupported AMD flash varlen boundary assertions')
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(tree), __file__ + ':varlen', 'exec'), namespace)
    return namespace['attention_forward_prefill_triton_impl']


def configure(*, max_num_seqs, max_model_len, max_num_batched_tokens, device,
              iu4_prefill_library=None):
    """Allocate one reusable arena, returning its accounted tensor bytes."""
    global _arena, _prefill
    device = torch.device(device)
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    library = (os.path.realpath(os.path.expanduser(os.fspath(iu4_prefill_library)))
               if iu4_prefill_library is not None else None)
    identity = (max_num_seqs, max_model_len, max_num_batched_tokens, device)
    if _arena is not None:
        if _arena['identity'] != identity or _arena['iu4_library'] != library:
            raise RuntimeError('Compact attention arena was configured differently')
        return _arena['bytes']
    if (not 1 <= max_num_seqs <= 8 or not 1 <= max_model_len <= 262144
            or not 1 <= max_num_batched_tokens <= 2048
            or torch.cuda.is_current_stream_capturing()):
        raise ValueError('Configure target compact prefill before capture, within C8/256K/2048 tokens')
    _prefill = _graph_safe_prefill()
    key = torch.empty((max_num_seqs * max_model_len, 2, 256),
                      dtype=torch.bfloat16, device=device)
    value = torch.empty_like(key)
    cu_k = torch.empty(max_num_seqs + 1, dtype=torch.int32, device=device)
    lse = torch.empty((16, max_num_batched_tokens), dtype=torch.float32, device=device)
    tensors = (key, value, cu_k, lse)
    iu4 = None
    if library is not None:
        from .attention_iu4 import NativeAttention
        iu4 = NativeAttention(device, library, max_model_len=max_model_len)
    size = sum(t.numel() * t.element_size() for t in tensors)
    if iu4 is not None:
        size += iu4.bytes
    _arena = dict(identity=identity, key=key, value=value, cu_k=cu_k, lse=lse,
                  bytes=size, iu4=iu4, iu4_library=library)
    return size


def try_forward(query, key_cache, value_cache, output, block_table,
                query_start_loc, seq_lens, *, max_query_len, max_seq_len, sm_scale):
    """Return False for an unsupported call; otherwise write output and return True.

    Supports variable-length C1..C8 prefill, mixed small queries, empty request
    slots and trailing graph token padding. CUDA metadata stays on the GPU.
    Target eligibility (causal, no window/sinks/bias/output scaling) is the
    caller's responsibility. The native cache writer must have run first.
    """
    page_size = key_cache.shape[3] if key_cache.ndim == 5 else 0
    if (max_query_len <= 8 or query.ndim != 3 or query.shape[1:] != (16, 256)
            or page_size not in (1120, 2240) or key_cache.shape[1:] != (2, 32, page_size, 8)
            or value_cache.ndim != 4 or value_cache.shape[1:] != (2, 256, page_size)
            or output.shape != query.shape or query.stride(2) != 1
            or output.stride(2) != 1 or query_start_loc.ndim != 1
            or seq_lens.ndim != 1 or block_table.ndim != 2
            or block_table.stride(1) != 1
            or not query_start_loc.is_contiguous() or not seq_lens.is_contiguous()
            or query_start_loc.numel() != seq_lens.numel() + 1
            or block_table.shape[0] < seq_lens.numel()
            or block_table.shape[1] * page_size < max_seq_len
            or any(t.dtype != torch.bfloat16 for t in (query, key_cache, value_cache, output))
            or any(t.dtype != torch.int32 for t in (block_table, query_start_loc, seq_lens))
            or any(t.device != query.device for t in (key_cache, value_cache, output,
                       block_table, query_start_loc, seq_lens))):
        return False
    if _arena is None:
        raise RuntimeError('Configure compact attention before memory profiling')
    max_reqs, max_length, max_tokens, device = _arena['identity']
    requests, rows = seq_lens.numel(), query.shape[0]
    if (not 1 <= requests <= max_reqs or not 1 <= max_seq_len <= max_length
            or not 1 <= rows <= max_tokens or query.device != device):
        return False
    # Capacity-sized views avoid reading the final cumulative GPU length.
    # The original flash kernel bounds every request by cu_q/cu_k instead.
    key = _arena['key'][:requests * max_seq_len]
    value = _arena['value'][:requests * max_seq_len]
    cu_k = _arena['cu_k'][:requests + 1]
    lse = _arena['lse'][:, :rows]
    _kv_boundaries[(1,)](query_start_loc, seq_lens, cu_k, REQUESTS=requests,
                         BLOCK=triton.next_power_of_2(requests), num_warps=1)
    _gather[(triton.cdiv(max_seq_len, 32), 2, requests)](
        key_cache, value_cache, block_table, cu_k, key, value,
        *key_cache.stride(), *value_cache.stride(),
        TABLE_STRIDE=block_table.stride(0), TOKENS=32, PAGE_SIZE=page_size,
        num_warps=8, num_stages=1)
    # Graph-padded rows are not real requests and receive defined zero output.
    output.zero_()
    if (_arena['iu4'] is not None and requests == 1 and max_query_len == 1120
            and max_seq_len >= 4096 and max_seq_len % 32 == 0
            and max_query_len <= rows and sm_scale == 256**-.5):
        _arena['iu4'].forward(query, key, value, output, query_start_loc,
                              seq_lens, max_seq_len)
        logger.info_once('Ornith C1 IU4 prefill active: BF16 cache gather + '
                         'normalized Q/K H256 and P/V H32, signed IU4 QK/PV; '
                         'Q1120 and aligned K>=4096, existing BF16 fallback elsewhere')
        return True
    _prefill(q=query, k=key, v=value, o=output, softmax_lse=lse,
        sd_mask=None, sm_scale=sm_scale, alibi_slopes=None, causal=True,
        window_size_left=-1, window_size_right=-1, bias=None, layout='thd',
        cu_seqlens_q=query_start_loc, cu_seqlens_k=cu_k,
        # Varlen reads actual Q/K lengths from cu_q/cu_k. These are only
        # launch bounds (Q also sets the rectangular grid), yet upstream
        # specializes both as constexpr. Stable arena bounds avoid compiling
        # a new flash kernel at every growing-context prefill chunk. Empty Q
        # tiles return before the attention loop in the installed kernel.
        max_seqlens_q=max_tokens, max_seqlens_k=max_length,
        dropout_p=0.0, philox_seed=0, philox_offset=0,
        return_scores=False, use_exp2=True,
        q_descale=None, k_descale=None, v_descale=None)
    logger.info_once('Ornith compact target prefill active: shared KV gather + AMD '
                     'Triton varlen flash; <=8-query decode unchanged')
    return True
