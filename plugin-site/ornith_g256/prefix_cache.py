"""Opt-in metadata bridge for the retained Ornith/DFlash2 prefix-cache layout.

The installed planner has no marker for Qwen's ordinary sliding-window draft
layers. Its conservative fallback marks target Mamba groups as draft groups too.
This model-scoped bridge labels the exact, separate draft group. Fine prefix
reuse optionally retains draft KV under full-attention allocation while keeping
the drafter's sliding attention window and existing accepted-state copying.
"""
# Copyright 2026 Ciru.
import logging


logger = logging.getLogger(__name__)
EXPECTED_DRAFT_NAMES = frozenset(
    f'model.layers.{index}.self_attn.attn' for index in range(40, 46)
)
_upstream_annotator = None
_upstream_mamba_split = None
_upstream_scheduler_init = None


def retain_draft_history(config):
    return config.additional_config.get('ornith_g256', {}).get('draft_full_retention', False)


def retained_draft_specs(config, specs):
    """Use the existing full-retention representation for six windowed layers.

    FullAttentionSpec.sliding_window explicitly preserves windowed computation
    while retaining all physical pages. That permits existing fine-grained
    full-attention/Mamba lookup and CoW without adding a new state cache.
    """
    from .cache_full1120 import padded_specs
    specs = padded_specs(config, specs)
    if not retain_draft_history(config):
        return specs
    from dataclasses import fields
    from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, SlidingWindowSpec
    validate_prefix_cache_config(config)
    result = dict(specs)
    names = {name for name in specs if name.startswith('model.layers.')}
    if names != EXPECTED_DRAFT_NAMES:
        raise ValueError('Fine prefix reuse requires exactly the six DFlash2 KV layers')
    for name in names:
        spec = specs[name]
        if (not isinstance(spec, SlidingWindowSpec) or spec.sliding_window != 4096
                or spec.extra_retained_tokens != 0):
            raise ValueError('Expected original DFlash2 4096-token window')
        result[name] = FullAttentionSpec(
            **{field.name: getattr(spec, field.name) for field in fields(AttentionSpec)},
            sliding_window=spec.sliding_window)
    return result


class _AttributeView:
    """Read-through object view; overrides never mutate the wrapped objects."""
    def __init__(self, wrapped, **overrides):
        object.__setattr__(self, '_wrapped', wrapped)
        object.__setattr__(self, '_overrides', overrides)

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        raise AttributeError('Alignment helper view is read-only')


def _mamba_block_aligned_split(self, request, num_new_tokens,
                               num_new_local_computed_tokens=0,
                               num_external_computed_tokens=0):
    config = self.vllm_config
    if (getattr(config.model_config, 'quantization', None) != 'ornith_g256'
            or not self.cache_config.enable_prefix_caching):
        return _upstream_mamba_split(self, request, num_new_tokens,
            num_new_local_computed_tokens, num_external_computed_tokens)
    spec = config.speculative_config
    if (self.block_size != 1120 or self.cache_config.mamba_cache_mode != 'align'
            or spec is None or spec.method != 'dflash' or spec.num_speculative_tokens != 15
            or self.scheduler_config.async_scheduling
            or not self.scheduler_config.enable_chunked_prefill):
        raise ValueError('Unexpected scheduler configuration for Ornith aligned prefix reuse')
    # EngineCore resets cache_config.block_size to the minimum participating
    # group size (560 for the draft), while self.block_size remains the resolved
    # LCM (1120 for Mamba/target). The upstream helper reads the former. Give
    # only this read-only helper the resolved alignment; keep the actual cache
    # config, hash granularity, allocation and every other helper rule intact.
    view = _AttributeView(self, cache_config=_AttributeView(
        self.cache_config, block_size=self.block_size))
    return _upstream_mamba_split(view, request, num_new_tokens,
        num_new_local_computed_tokens, num_external_computed_tokens)


def _install_mamba_alignment():
    # Cache-group annotation happens after model/config imports are complete.
    # Importing the scheduler here avoids a plugin-registration import cycle.
    from vllm.v1.core.sched.scheduler import Scheduler
    global _upstream_mamba_split
    current = Scheduler._mamba_block_aligned_split
    if current is _mamba_block_aligned_split:
        return
    if _upstream_mamba_split is None:
        _upstream_mamba_split = current
    elif current is not _upstream_mamba_split:
        raise RuntimeError('Another extension replaced Mamba prefill alignment')
    Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split


def _scheduler_init(self, *args, **kwargs):
    # The upstream constructor creates the coordinator and all managers. No
    # request has been admitted yet, so its derived lookup policy can be set
    # coherently here while the model's VllmConfig is still directly available.
    _upstream_scheduler_init(self, *args, **kwargs)
    config = self.vllm_config
    if (getattr(config.model_config, 'quantization', None) != 'ornith_g256'
            or not config.cache_config.enable_prefix_caching):
        return
    if config.cache_config.block_size not in (560, 1120):
        raise ValueError('Unexpected physical block size for Ornith DFlash prefix reuse')
    validate_prefix_cache_config(_AttributeView(config, cache_config=_AttributeView(
        config.cache_config, block_size=self.block_size)))
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    coordinator = self.kv_cache_manager.coordinator
    groups = coordinator.kv_cache_config.kv_cache_groups
    # Reuse the full model/layout validation, including all target names/types.
    # Re-installing the hooks is idempotent; the config view exposes the resolved
    # scheduler alignment instead of EngineCore's minimum physical group size.
    validated_config = _AttributeView(config, cache_config=_AttributeView(
        config.cache_config, block_size=self.block_size))
    specs = {name: group.kv_cache_spec for group in groups for name in group.layer_names}
    annotate_draft_group(validated_config, specs, groups)
    if (not isinstance(coordinator, HybridKVCacheCoordinator)
            or coordinator.num_reprefillable_tokens != 0):
        raise ValueError('Expected hybrid DFlash with no multi-module re-prefill tail')
    draft_ids = [i for i, group in enumerate(groups)
                 if set(group.layer_names) == EXPECTED_DRAFT_NAMES]
    if len(draft_ids) != 1 or coordinator.eagle_group_ids != set(draft_ids):
        raise ValueError('Unexpected coordinator draft-group identity')
    draft_id = draft_ids[0]
    matching = [(i, group) for i, group in enumerate(coordinator.attention_groups)
                if draft_id in group.group_ids]
    expected_type = FullAttentionSpec if retain_draft_history(config) else SlidingWindowSpec
    if (len(matching) != 1 or matching[0][1].group_ids != [draft_id]
            or not isinstance(matching[0][1].spec, expected_type)):
        raise ValueError('Expected one exclusive DFlash lookup group')
    index, lookup_group = matching[0]
    # DFlash context KV uses original target positions; query KV starts at the
    # next accepted position. Unlike Eagle, no next-token embedding is written
    # into the final accepted context slot. Remove the lookup margin AND drop
    # together, and make cache retention use the same unshifted window policy.
    coordinator.attention_groups[index] = lookup_group._replace(use_eagle=False)
    coordinator.single_type_managers[draft_id].use_eagle = False
    # Keep is_eagle_group/eagle_group_ids: they identify the draft and prevent
    # the upstream conservative fallback from marking target groups as drafts.
    logger.info('Ornith prefix reuse: DFlash context KV uses unshifted lookup; '
                'disabled only its Eagle lookahead margin and last-block drop')
    if not retain_draft_history(config):
        # Isolated diagnostic: default GCD hashing otherwise auto-enables
        # partial Mamba tails even with prefix_match_unit=None.
        coordinator.enable_partial_hash_hits = False
        self.mamba_partial_cache_hit = False
        assert self.block_size == coordinator.scheduler_block_size == 1120
        assert coordinator._cache_hit_alignment_tokens == 1120
        assert coordinator._align_cacheable(3360) == 3360
        assert all(manager.scheduler_block_size == 1120
                   for manager in coordinator.single_type_managers)
        logger.info('Ornith full-boundary diagnostic: hash=%s scheduler=%s '
                    'hit_alignment=%s cacheable3360=%s partial_hits=%s '
                    'partial_prompt_stops=%s group_blocks=%s',
                    coordinator.hash_block_size, coordinator.scheduler_block_size,
                    coordinator._cache_hit_alignment_tokens,
                    coordinator._align_cacheable(3360),
                    coordinator.enable_partial_hash_hits, self.mamba_partial_cache_hit,
                    [manager.block_size for manager in coordinator.single_type_managers])
    if retain_draft_history(config):
        if not (coordinator.enable_partial_hash_hits and self.mamba_partial_cache_hit
                and coordinator.hash_block_size == 8):
            raise ValueError('Fine prefix lookup and Mamba prompt-tail checkpoints were not enabled')
        logger.info('Ornith fine prefix reuse: retained draft KV, unchanged 4096-token '
                    'attention window, match unit8, existing Mamba prompt-tail CoW')


def _install_dflash_lookup():
    from vllm.v1.core.sched.scheduler import Scheduler
    global _upstream_scheduler_init
    current = Scheduler.__init__
    if current is _scheduler_init:
        return
    if _upstream_scheduler_init is None:
        _upstream_scheduler_init = current
    elif current is not _upstream_scheduler_init:
        raise RuntimeError('Another extension replaced scheduler construction')
    Scheduler.__init__ = _scheduler_init


def validate_prefix_cache_config(config):
    """Return whether this is the supported experiment, rejecting other opt-ins."""
    if not config.cache_config.enable_prefix_caching:
        return False
    spec = config.speculative_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    if (spec is None or spec.method != 'dflash' or spec.num_speculative_tokens != 15
            or cache.mamba_cache_mode != 'align' or cache.block_size != 1120
            or scheduler.async_scheduling or not scheduler.enable_chunked_prefill):
        raise ValueError('Ornith prefix reuse requires DFlash15, Mamba align, KV block1120, '
                         'synchronous scheduling and chunked prefill')
    if retain_draft_history(config):
        if cache.prefix_match_unit != 8:
            raise ValueError('Fine prefix reuse requires prefix_match_unit8')
    elif cache.prefix_match_unit is not None:
        raise ValueError('Finer prefix matching requires retained DFlash history')
    return True


def annotate_draft_group(config, kv_cache_spec, groups):
    """Validate source-derived names/types before marking the one draft group."""
    if (getattr(config.model_config, 'quantization', None) != 'ornith_g256'
            or not config.cache_config.enable_prefix_caching):
        return
    validate_prefix_cache_config(config)
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

    target = config.model_config
    draft = config.speculative_config.draft_model_config
    if (target.architecture != 'Qwen3_5MoeForConditionalGeneration'
            or target.hf_text_config.num_hidden_layers != 40
            or draft.architecture != 'DFlash2DraftModel'
            or draft.hf_text_config.num_hidden_layers != 6):
        raise ValueError('Ornith prefix reuse only supports the retained40-layer target and6-layer DFlash2 draft')
    actual_draft_names = {name for name in kv_cache_spec if name.startswith('model.layers.')}
    if actual_draft_names != EXPECTED_DRAFT_NAMES:
        raise ValueError('Unexpected Ornith draft KV layer names; refusing ambiguous prefix-cache annotation')
    expected_type = FullAttentionSpec if retain_draft_history(config) else SlidingWindowSpec
    if any(not isinstance(kv_cache_spec[name], expected_type)
           or kv_cache_spec[name].sliding_window != 4096 for name in EXPECTED_DRAFT_NAMES):
        raise ValueError('Expected six DFlash KV specs with the original4096-token window')
    target_names = set(kv_cache_spec) - EXPECTED_DRAFT_NAMES
    if (len(target_names) != 40
            or any(not name.startswith('language_model.model.layers.') for name in target_names)
            or sum(isinstance(kv_cache_spec[name], MambaSpec) for name in target_names) != 30
            or sum(isinstance(kv_cache_spec[name], FullAttentionSpec) for name in target_names) != 10):
        raise ValueError('Unexpected Ornith target KV layout; expected30GDN and10attention layers')
    containing = [group for group in groups if EXPECTED_DRAFT_NAMES.intersection(group.layer_names)]
    if (len(containing) != 1 or set(containing[0].layer_names) != EXPECTED_DRAFT_NAMES
            or not isinstance(containing[0].kv_cache_spec, expected_type)):
        raise ValueError('Ornith prefix reuse requires all six draft layers in one exclusive group')
    if any(group.is_eagle_group for group in groups if group is not containing[0]):
        raise ValueError('Upstream marked an Ornith target group as a drafter; refusing conflicting annotation')
    containing[0].is_eagle_group = True
    _install_mamba_alignment()
    _install_dflash_lookup()
    from .prefill_draft import install as install_prefill_sampling
    install_prefill_sampling()
    logger.info('Ornith prefix reuse: marked exactly six DFlash2 sliding-window layers as the draft group')


def _annotate_eagle_groups(vllm_config, kv_cache_spec, kv_cache_groups, use_deepseek_v4_fallback=False):
    # Preserve every installed upstream annotation rule before applying ours.
    _upstream_annotator(vllm_config, kv_cache_spec, kv_cache_groups,
                        use_deepseek_v4_fallback=use_deepseek_v4_fallback)
    annotate_draft_group(vllm_config, kv_cache_spec, kv_cache_groups)


def install():
    """Patch the planner's module-global hook once, without installed-file edits."""
    from vllm.v1.core import kv_cache_utils
    global _upstream_annotator
    current = kv_cache_utils._annotate_eagle_groups
    if current is _annotate_eagle_groups:
        return
    if _upstream_annotator is None:
        _upstream_annotator = current
    elif current is not _upstream_annotator:
        raise RuntimeError('Another extension replaced KV draft-group annotation')
    kv_cache_utils._annotate_eagle_groups = _annotate_eagle_groups
