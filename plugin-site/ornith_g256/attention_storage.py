# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru.
"""Store target KV token-major inside the existing vLLM cache pages.

Allocation, block IDs, prefix ownership and cache size remain backend-owned.
The writer receives true BSHD views; stride-aware attention receives the
equivalent historical 5D/4D views. No second cache or history copy is created.
"""
import types

import torch
from vllm.logger import init_logger

logger = init_logger('vllm.ornith_g256.attention_storage')
_installed = False


def _flash_views(cache):
    # Bound backend cache is [blocks, K/V, tokens, heads * dim].
    page_size = cache.shape[2] if cache.ndim == 4 else 0
    if (page_size not in (1120, 2240) or cache.shape[1:] != (2, page_size, 512)
            or cache.dtype != torch.bfloat16
            or cache.stride(2) != 512 or cache.stride(3) != 1):
        raise ValueError('Token-major target KV requires BF16 [B,2,page,512], page1120/2240')
    blocks = cache.shape[0]
    return (cache[:, 0].view(blocks, page_size, 2, 256),
            cache[:, 1].view(blocks, page_size, 2, 256))


def _split_for_attention(cache, num_heads, head_size):
    # RocmAttentionImpl passes the K/V-first transpose to this seam.
    if (num_heads, head_size) != (2, 256):
        raise ValueError('Token-major cache views are target-specific')
    key, value = _flash_views(cache.transpose(0, 1))
    blocks, page_size = key.shape[:2]
    key_view = key.as_strided((blocks, 2, 32, page_size, 8),
                             (key.stride(0), 256, 8, 512, 1))
    value_view = value.as_strided((blocks, 2, 256, page_size),
                                 (value.stride(0), 256, 1, 512))
    return key_view, value_view


def install():
    """Install only in an isolated worker before loading/capturing the model."""
    global _installed
    if _installed:
        return
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends import rocm_attn
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    if rocm_aiter_ops.is_enabled():
        raise ValueError('Token-major KV currently uses the unfused ROCm cache writer')
    impl = rocm_attn.RocmAttentionImpl
    original_forward = impl.forward
    original_update = impl.do_kv_cache_update
    namespace = dict(original_forward.__globals__)
    namespace['PagedAttention'] = types.SimpleNamespace(
        split_kv_cache=_split_for_attention)
    forward_with_views = types.FunctionType(
        original_forward.__code__, namespace, 'forward_token_major',
        original_forward.__defaults__, original_forward.__closure__)
    forward_with_views.__kwdefaults__ = original_forward.__kwdefaults__

    def target(self):
        return (self.attn_type == AttentionType.DECODER
                and (self.num_heads, self.num_kv_heads, self.head_size) == (16, 2, 256)
                and self.kv_cache_dtype == 'auto'
                and self.alibi_slopes is None and self.sinks is None
                and self.sliding_window == (-1, -1)
                and self.logits_soft_cap == 0)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        selected = forward_with_views if target(self) else original_forward
        return selected(self, layer, query, key, value, kv_cache, attn_metadata,
                        output, output_scale, output_block_scale)

    def update(self, layer, key, value, kv_cache, slot_mapping):
        if not target(self):
            return original_update(self, layer, key, value, kv_cache, slot_mapping)
        key_cache, value_cache = _flash_views(kv_cache)
        triton_reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping,
            self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        logger.info_once('Ornith target KV uses token-major pages in the existing cache allocation')

    impl.forward = forward
    impl.do_kv_cache_update = update
    _installed = True
