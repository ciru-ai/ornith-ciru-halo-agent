"""Isolated full-state1120 cache geometry; original tensors and arithmetic."""
import logging
from dataclasses import replace

PAGE_BYTES = 2_392_064
TARGET_BLOCK = 1120
DRAFT_BLOCK = 560
_upstream_align = None
logger = logging.getLogger(__name__)


def enabled(config):
    return (getattr(config.model_config, 'quantization', None) == 'ornith_g256'
            and config.cache_config.enable_prefix_caching
            and config.additional_config.get('ornith_g256', {}).get('dynamic_spec_profile', False))


def _align(cls, vllm_config, backend_cls):
    _upstream_align(cls, vllm_config, backend_cls)
    if not enabled(vllm_config):
        return
    cache = vllm_config.cache_config
    spec = vllm_config.speculative_config
    assert cache.mamba_cache_mode == 'align' and cache.prefix_match_unit is None
    assert spec.method == 'dflash' and spec.num_speculative_tokens == 15
    cache.block_size = TARGET_BLOCK
    cache.mamba_block_size = TARGET_BLOCK
    cache.mamba_page_size_padded = PAGE_BYTES
    logger.info('Ornith full1120 platform: target/Mamba=%s draft=%s physical_page=%s',
                TARGET_BLOCK, DRAFT_BLOCK, PAGE_BYTES)


def padded_specs(config, specs):
    if not enabled(config):
        return specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
    from .prefix_cache import EXPECTED_DRAFT_NAMES
    drafts = {name for name in specs if name.startswith('model.layers.')}
    targets = set(specs) - drafts
    assert drafts == EXPECTED_DRAFT_NAMES
    assert len(targets) == 40 and all(name.startswith('language_model.model.layers.') for name in targets)
    assert sum(isinstance(specs[name], MambaSpec) for name in targets) == 30
    assert sum(isinstance(specs[name], FullAttentionSpec) for name in targets) == 10
    result = {}
    for name, spec in specs.items():
        if name in drafts:
            assert isinstance(spec, SlidingWindowSpec) and spec.sliding_window == 4096
            block = DRAFT_BLOCK
        else:
            block = TARGET_BLOCK
        if isinstance(spec, MambaSpec):
            assert spec.real_page_size_bytes == PAGE_BYTES
        updated = replace(spec, block_size=block, page_size_padded=PAGE_BYTES)
        assert updated.page_size_bytes == PAGE_BYTES
        result[name] = updated
    logger.info('Ornith full1120 specs: target_attention=10x1120 Mamba=30x1120 '
                'draft=6x560 all_page_bytes=%s recurrent_dtype=%s', PAGE_BYTES,
                next(spec.dtypes for spec in result.values() if isinstance(spec, MambaSpec)))
    return result


def install():
    from vllm.platforms.interface import Platform
    global _upstream_align
    current = Platform._align_hybrid_block_size.__func__
    if current is _align:
        return
    if _upstream_align is None:
        _upstream_align = current
    elif current is not _upstream_align:
        raise RuntimeError('Another extension replaced hybrid cache alignment')
    Platform._align_hybrid_block_size = classmethod(_align)
