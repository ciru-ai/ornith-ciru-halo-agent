"""Avoid sampling DFlash candidates for scheduler-discarded prefill rows.

Unfinished chunked-prefill requests have no valid sampled anchor token. Their
draft query rows are placeholders, not real decode predictions. The original
proposer still runs every forward/context-KV update; only the subsequent
candidate-head/selector call omits requests already marked for discard.
"""
# Copyright 2026 Ciru.
import dataclasses
import torch


_upstream_propose = None
_upstream_sample = None
_MISSING = object()
_MASK_ATTR = '_ornith_discard_prefill_requests'


def _propose_draft_token_ids(self, *args, **kwargs):
    config = self.vllm_config
    spec = config.speculative_config
    if (getattr(config.model_config, 'quantization', None) != 'ornith_g256'
            or not config.cache_config.enable_prefix_caching
            or config.cache_config.mamba_cache_mode != 'align'
            or spec is None or spec.method != 'dflash' or spec.num_speculative_tokens != 15
            or not getattr(self.drafter, 'is_dflash2', False)
            or spec.disable_padded_drafter_batch):
        return _upstream_propose(self, *args, **kwargs)
    if config.scheduler_config.async_scheduling:
        raise ValueError('Ornith prefill sampling scope requires synchronous scheduling')
    # This is the same existing CPU mask used by _is_all_reqs_chunked_prefill;
    # there is no device-to-host read or inference from hidden-state values.
    mask = tuple(bool(value) for value in
                 self.discard_request_mask.np[:self.input_batch.num_reqs])
    prior = getattr(self.drafter, _MASK_ATTR, _MISSING)
    setattr(self.drafter, _MASK_ATTR, mask)
    try:
        return _upstream_propose(self, *args, **kwargs)
    finally:
        if prior is _MISSING:
            delattr(self.drafter, _MASK_ATTR)
        else:
            setattr(self.drafter, _MASK_ATTR, prior)


def _sample_draft_tokens(self, hidden_states, sampling_metadata):
    discarded = getattr(self, _MASK_ATTR, None)
    if not self.is_dflash2 or discarded is None or not any(discarded):
        return _upstream_sample(self, hidden_states, sampling_metadata)
    anchors = self._dflash_anchor_token_ids
    if anchors is None:
        raise ValueError('DFlash2 candidate sampling has no anchor metadata')
    batch_size = anchors.numel()
    steps = self.num_speculative_tokens
    if (len(discarded) != batch_size or hidden_states.ndim != 2
            or hidden_states.shape[0] != batch_size * steps):
        raise ValueError('Discard mask does not match DFlash2 request-major candidate rows')
    active = [index for index, discard in enumerate(discarded) if not discard]
    if not active:
        # These IDs/probability rows are never accepted by the scheduler. Keep
        # the original flattened B*K interface without invoking head/selector.
        ids = torch.zeros(batch_size * steps, dtype=torch.long, device=hidden_states.device)
        probs = (None if sampling_metadata.all_greedy else torch.zeros(
            batch_size * steps, int(self.draft_model_config.hf_config.vocab_size),
            dtype=torch.float32, device=hidden_states.device))
        return ids, probs

    rows = torch.tensor(active, dtype=torch.long, device=hidden_states.device)
    compact_hidden = hidden_states.reshape(batch_size, steps, -1).index_select(
        0, rows).reshape(len(active) * steps, -1)
    metadata = sampling_metadata
    if sampling_metadata.temperature is not None:
        # The DFlash2 sampler reads only all_greedy and per-request temperature;
        # preserving all_greedy also preserves its probability-return contract.
        metadata = dataclasses.replace(sampling_metadata,
            temperature=sampling_metadata.temperature[:batch_size].index_select(0, rows))
    self._dflash_anchor_token_ids = anchors.index_select(0, rows)
    try:
        active_ids, active_probs = _upstream_sample(self, compact_hidden, metadata)
    finally:
        self._dflash_anchor_token_ids = anchors
    if active_ids.numel() != len(active) * steps:
        raise ValueError('Unexpected compact DFlash2 candidate-ID shape')
    ids = active_ids.new_zeros((batch_size, steps))
    ids.index_copy_(0, rows, active_ids.reshape(len(active), steps))
    probs = None
    if active_probs is not None:
        if active_probs.ndim != 2 or active_probs.shape[0] != len(active) * steps:
            raise ValueError('Unexpected compact DFlash2 probability shape')
        probs = active_probs.new_zeros((batch_size, steps, active_probs.shape[-1]))
        probs.index_copy_(0, rows, active_probs.reshape(len(active), steps, -1))
        probs = probs.flatten(0, 1).contiguous()
    return ids.reshape(-1), probs


def install():
    """Install lazily from the validated Ornith prefix-cache annotation path."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.spec_decode.dflash import DFlashProposer
    global _upstream_propose, _upstream_sample
    current_propose = GPUModelRunner.propose_draft_token_ids
    current_sample = DFlashProposer._sample_draft_tokens
    if current_propose is _propose_draft_token_ids and current_sample is _sample_draft_tokens:
        return
    if _upstream_propose is None and _upstream_sample is None:
        _upstream_propose, _upstream_sample = current_propose, current_sample
    elif current_propose is not _upstream_propose or current_sample is not _upstream_sample:
        raise RuntimeError('Another extension replaced DFlash proposal or sampling')
    GPUModelRunner.propose_draft_token_ids = _propose_draft_token_ids
    DFlashProposer._sample_draft_tokens = _sample_draft_tokens
