"""Bound long-prefill interference using the pinned scheduler's native deferral.

The September Apodex policy supplies the deferral mechanism. Ornith's concurrent
K0 path needs a larger decode service window and a bounded total prefill chunk.
All token reservation, KV allocation, output sampling and request completion
remain in the native scheduler. Pure prefill and short prompts keep its policy.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class Configuration:
    cadence: int = 16
    long_threshold: int = 8192
    mixed_chunk_tokens: int = 1120

    def __post_init__(self):
        if not 1 <= self.cadence <= 64:
            raise ValueError('CIRU_PREFILL_CADENCE must be 1..64')
        if self.long_threshold < 0:
            raise ValueError('CIRU_PREFILL_LONG_THRESHOLD must be nonnegative')
        if self.mixed_chunk_tokens not in (1120, 4480):
            raise ValueError('CIRU_PREFILL_MIXED_CHUNK_TOKENS must be 1120 or 4480')


@lru_cache(maxsize=1)
def configuration():
    return Configuration(
        cadence=int(os.environ.get('CIRU_PREFILL_CADENCE', '16')),
        long_threshold=int(os.environ.get('CIRU_PREFILL_LONG_THRESHOLD', '8192')),
        mixed_chunk_tokens=int(os.environ.get('CIRU_PREFILL_MIXED_CHUNK_TOKENS', '1120')),
    )


def schedule(original, scheduler, throttle_prefills=False):
    config = configuration()
    enabled = scheduler.vllm_config.additional_config.get('ornith_g256', {}).get('adaptive_c1_fallback')
    if config.cadence == 1 or not enabled:
        return original(scheduler, throttle_prefills=throttle_prefills)
    # Use the same decoder predicate as the pinned native deferral mechanism.
    decoders = tuple(r for r in scheduler.running if not r.is_prefill_chunk)
    long_pending = any(
        not r.is_finished()
        and r.num_prompt_tokens > config.long_threshold
        and r.num_computed_tokens < r.num_prompt_tokens
        and not scheduler._is_blocked_waiting_status(r.status)
        for r in scheduler.requests.values()
    )
    if not decoders or not long_pending:
        scheduler._ornith_prefill_service_step = 0
        return original(scheduler, throttle_prefills=throttle_prefills)

    # Admit a bounded chunk immediately, then grant cadence-1 decode-only turns.
    # This per-scheduler counter is independent of earlier idle/prefill steps.
    step = getattr(scheduler, '_ornith_prefill_service_step', 0)
    defer = step % config.cadence != 0
    scheduler._ornith_prefill_service_step = step + 1
    settings = scheduler.scheduler_config
    previous_threshold = settings.long_prefill_token_threshold
    previous_capacity_bound = scheduler.prefill_capacity_bound
    previous_budget = scheduler.max_num_scheduled_tokens
    if previous_threshold > 0:
        settings.long_prefill_token_threshold = min(previous_threshold, config.mixed_chunk_tokens)
    else:
        settings.long_prefill_token_threshold = config.mixed_chunk_tokens
    # The concurrent K0 adapter clears speculative IDs before reservation.
    # Reserve one row per decoder plus ONE total prefill block, including when
    # several long prefills are queued. Native Mamba alignment handles the tail.
    scheduler.max_num_scheduled_tokens = min(previous_budget,
        config.mixed_chunk_tokens + len(decoders))
    if defer:
        scheduler.prefill_capacity_bound = False
    signature = (config.cadence, config.long_threshold, config.mixed_chunk_tokens)
    if getattr(scheduler, '_ornith_prefill_service_logged', None) != signature:
        print('ORNITH_PREFILL_SERVICE ' + json.dumps(dict(
            pid=os.getpid(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cadence=config.cadence, long_threshold=config.long_threshold,
            mixed_chunk_tokens=config.mixed_chunk_tokens,
            total_prefill_budget_bounded=True, native_deferral=True,
            original_long_prefill_token_threshold=previous_threshold,
            original_max_num_scheduled_tokens=previous_budget,
            pure_prefill_policy_unchanged=True, concurrent_draft_depth=0,
        )), flush=True)
        scheduler._ornith_prefill_service_logged = signature
    try:
        return original(scheduler, throttle_prefills=throttle_prefills or defer)
    finally:
        settings.long_prefill_token_threshold = previous_threshold
        scheduler.max_num_scheduled_tokens = previous_budget
        if defer:
            scheduler.prefill_capacity_bound = previous_capacity_bound
