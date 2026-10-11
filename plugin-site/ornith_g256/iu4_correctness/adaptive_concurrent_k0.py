"""Optional source-bound K0 contingency for multiple unfinished requests.

Install on the frozen fallback module BEFORE native package import invokes its
install(). The pinned synchronous scheduler clears only speculative ID lists
before reservation; its committed tokens, KV positions and capacity stay native.
The guard is conservative: a waiting second request also suppresses speculation
on an actual C1 step. True sole-request C1 keeps the frozen policy unchanged.
Fresh concurrent admissions bypass prefix-cache reads and retain that choice
for fresh recomputation; ongoing committed KV/state is never reset. This is a
cache-read isolation workaround, not a repair of shared checkpoint ownership.
This module is not selected or installed by the final launcher by default.
"""
from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path


POLICY_SHA256 = '6e8be45ea6cee7b995009f967f08190a32b18327a42fd768b252faa875dbed5e'
SCHEDULER_SHA256 = '30160d4df0366e22f9ff583d991e2898b3b65c5d25e471a209578336e24001b5'
REQUEST_SHA256 = 'da610d5203e72d2d524adc5fb39df11502a23520f2f31d9a44cf13d343f9c25d'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _enabled(scheduler):
    return bool(scheduler.vllm_config.additional_config.get('ornith_g256', {}).get('adaptive_c1_fallback'))


def _live(scheduler):
    return tuple(request for request in scheduler.requests.values() if not request.is_finished())


def _check_runtime(scheduler, requests):
    if (scheduler.scheduler_config.async_scheduling or scheduler.use_pp
            or scheduler.use_v2_model_runner or scheduler.max_num_running_reqs != 8
            or getattr(scheduler.vllm_config, 'max_concurrent_batches', None) != 1):
        raise RuntimeError('Concurrent K0 supports only synchronous V1 single-rank C1-8 with one concurrent batch')
    if any(request.num_output_placeholders or request.num_in_flight_tokens for request in requests):
        raise RuntimeError('Concurrent K0 cannot clear drafts for an in-flight or placeholder request')
    if any(type(request.spec_token_ids) is not list for request in requests):
        raise RuntimeError('Concurrent K0 expects native speculative ID lists')
    return dict(max_concurrent_batches=scheduler.vllm_config.max_concurrent_batches,
        async_scheduling=scheduler.scheduler_config.async_scheduling,
        pipeline_parallel=scheduler.use_pp, use_v2_model_runner=scheduler.use_v2_model_runner,
        max_num_running_reqs=scheduler.max_num_running_reqs,
        unfinished_requests_zero_in_flight_tokens=True,
        unfinished_requests_zero_output_placeholders=True)


def _clear(requests):
    removed = sum(len(request.spec_token_ids) for request in requests)
    for request in requests:
        request.spec_token_ids.clear()
    return removed


def _emit_install(state):
    receipt = dict(state['receipt'], pid=os.getpid())
    if state.get('runtime_config_pid') != receipt['pid']:
        receipt.update(runtime_config_verified=False, runtime_config=None)
    signature = (receipt['pid'], receipt['runtime_source_verified'], receipt['runtime_config_verified'])
    if state.get('emitted_install') != signature:
        print('ORNITH_CONCURRENCY_K0_INSTALLED ' + json.dumps(receipt), flush=True)
        state['emitted_install'] = signature
    return receipt


def _bind_scheduler(module, state):
    native = importlib.import_module('vllm.v1.core.sched.scheduler')
    requests = importlib.import_module('vllm.v1.request')
    if _sha(native.__file__) != SCHEDULER_SHA256 or _sha(requests.__file__) != REQUEST_SHA256:
        raise RuntimeError('Concurrent K0 pinned scheduler/request source differs')
    cls = native.Scheduler
    if state.get('runtime_cls') is not None:
        if (state['runtime_cls'] is not cls or cls.schedule is not state['schedule_wrapper']
                or cls.update_draft_token_ids is not state['draft_wrapper']
                or cls._get_local_prefix_cache_hit is not state['prefix_wrapper']):
            raise RuntimeError('Concurrent K0 installed scheduler wrapper changed')
        _emit_install(state)
        return
    if getattr(module, '_INSTALLED', False):
        raise RuntimeError('Concurrent K0 must precede frozen policy scheduler installation')
    original = cls.schedule
    original_prefix = cls._get_local_prefix_cache_hit
    if Path(original.__code__.co_filename).resolve() != Path(native.__file__).resolve():
        raise RuntimeError('Concurrent K0 scheduler schedule is already interposed')
    if Path(original_prefix.__code__.co_filename).resolve() != Path(native.__file__).resolve():
        raise RuntimeError('Concurrent K0 native prefix lookup is already interposed')

    @wraps(original_prefix)
    def prefix_read(scheduler, request, *args, **kwargs):
        live = _live(scheduler) if _enabled(scheduler) else ()
        latched = getattr(request, '_ornith_concurrent_prefix_read_isolated', False)
        if request.num_computed_tokens != 0 or not live or not (len(live) > 1 or latched):
            return original_prefix(scheduler, request, *args, **kwargs)
        if scheduler.connector is not None:
            raise RuntimeError('Concurrent prefix-read isolation does not support a KV connector')
        request.skip_reading_prefix_cache = True
        request._ornith_concurrent_prefix_read_isolated = True
        payload = dict(pid=os.getpid(), adapter_sha256=state['adapter_hash'],
            policy_source_sha256=POLICY_SHA256, scheduler_source_sha256=SCHEDULER_SHA256,
            request_source_sha256=REQUEST_SHA256, request_id=request.request_id,
            request_ids=tuple(row.request_id for row in live), unfinished_requests=len(live),
            num_computed_tokens=request.num_computed_tokens, fresh_only=True,
            prefix_read_bypassed=True, skip_reading_prefix_cache=True,
            empty_computed_blocks=True, num_local_computed_tokens=0,
            shared_prefix_boundary=0, hit_diverged=False,
            latch_for_fresh_recompute=True, previously_latched=latched,
            ongoing_committed_kv_reset=False, allocation_and_cache_writes_unchanged=True)
        print('ORNITH_CONCURRENCY_PREFIX_READ_BYPASS ' + json.dumps(payload), flush=True)
        return scheduler.kv_cache_manager.empty_kv_cache_blocks, 0, 0, False

    @wraps(original)
    def pre_reservation(scheduler, *args, **kwargs):
        live = _live(scheduler) if _enabled(scheduler) else ()
        guard = len(live) > 1
        scheduler._ornith_concurrent_k0_guard = guard
        scheduler._ornith_concurrent_k0_cleared = 0
        fresh_latched = any(request.num_computed_tokens == 0
            and getattr(request, '_ornith_concurrent_prefix_read_isolated', False) for request in live)
        if guard or fresh_latched:
            config = _check_runtime(scheduler, live)
            if scheduler.connector is not None and any(request.num_computed_tokens == 0 for request in live):
                raise RuntimeError('Concurrent prefix-read isolation does not support a KV connector')
            state['receipt'].update(runtime_config_verified=True, runtime_config=config)
            state['runtime_config_pid'] = os.getpid()
            _emit_install(state)
        if not guard:
            return original(scheduler, *args, **kwargs)
        scheduler._ornith_concurrent_k0_cleared = _clear(live)
        # The pinned scheduler can otherwise pad a newly resumed decode with
        # artificial [-1] draft rows. This scalar controls that padding only
        # during reservation; configured lookahead/capacity remain unchanged.
        capacity = scheduler.num_spec_tokens
        scheduler.num_spec_tokens = 0
        try:
            return original(scheduler, *args, **kwargs)
        finally:
            scheduler.num_spec_tokens = capacity

    cls._get_local_prefix_cache_hit = prefix_read
    cls.schedule = pre_reservation
    state['original_install']()
    frozen_schedule = cls.schedule
    frozen_draft = cls.update_draft_token_ids
    state['receipt']['runtime_source_verified'] = True

    @wraps(frozen_schedule)
    def schedule(scheduler, *args, **kwargs):
        _emit_install(state)
        from .prefill_service import schedule as schedule_with_prefill_service
        output = schedule_with_prefill_service(frozen_schedule, scheduler, *args, **kwargs)
        ids = tuple(output.num_scheduled_tokens)
        active = _enabled(scheduler)
        guard = getattr(scheduler, '_ornith_concurrent_k0_guard', False)
        if active and ids and (guard or 2 <= len(ids) <= 8):
            scheduled = sum(len(tokens) for tokens in output.scheduled_spec_decode_tokens.values())
            if scheduled or output.num_spec_tokens_to_schedule != 0:
                # Fail before handing an unsupported speculative step to the
                # worker. Never try to repair optimistic counts after reserve.
                raise RuntimeError('Concurrent K0 reservation admitted speculative rows')
            signature = (len(ids), guard)
            if getattr(scheduler, '_ornith_concurrent_k0_logged_mode', None) != signature:
                payload = dict(pid=os.getpid(), adapter_sha256=state['adapter_hash'],
                    policy_source_sha256=POLICY_SHA256,
                    scheduled_requests=len(ids), unfinished_requests=len(_live(scheduler)),
                    request_ids=ids, scheduled_draft_tokens=scheduled,
                    future_draft_depth=output.num_spec_tokens_to_schedule,
                    current_step_k0=True, reason='multiple_unfinished_requests',
                    queued_draft_ids_cleared=getattr(scheduler, '_ornith_concurrent_k0_cleared', 0),
                    committed_tokens_or_kv_positions_changed=False,
                    conservative_guard=guard, sampling_rng_changed=False)
                print('ORNITH_CONCURRENCY_K0_SCHEDULE ' + json.dumps(payload), flush=True)
                scheduler._ornith_concurrent_k0_logged_mode = signature
        else:
            scheduler._ornith_concurrent_k0_logged_mode = None
        return output

    @wraps(frozen_draft)
    def draft(scheduler, *args, **kwargs):
        result = frozen_draft(scheduler, *args, **kwargs)
        if _enabled(scheduler):
            live = _live(scheduler)
            if len(live) > 1:
                _clear(live)
        return result

    cls.schedule = schedule
    cls.update_draft_token_ids = draft
    state.update(runtime_cls=cls, schedule_wrapper=schedule, draft_wrapper=draft,
                 prefix_wrapper=prefix_read)
    _emit_install(state)


def install(module):
    """Prepare the optional interposer; native bootstrap verifies runtime files."""
    if _sha(module.__file__) != POLICY_SHA256:
        raise RuntimeError('Concurrent K0 requires the exact frozen fallback policy')
    adapter_hash = _sha(__file__)
    cls = module.Policy
    existing = cls.__dict__.get('_ornith_concurrent_k0_install')
    if existing is not None:
        if (existing['adapter_hash'] != adapter_hash or cls.budget is not existing['budget_wrapper']
                or module.install is not existing['install_wrapper']):
            raise RuntimeError('Concurrent K0 installed wrapper/source changed')
        return _emit_install(existing)
    if getattr(module, '_INSTALLED', False):
        raise RuntimeError('Concurrent K0 must precede frozen policy installation')
    original_budget = cls.budget
    original_install = module.install
    if original_budget.__module__ != module.__name__ or original_install.__module__ != module.__name__:
        raise RuntimeError('Concurrent K0 frozen policy methods are already interposed')
    receipt = dict(pid=os.getpid(), adapter_sha256=adapter_hash,
        policy_module=module.__name__, policy_source_sha256=POLICY_SHA256,
        scheduler_source_sha256=SCHEDULER_SHA256, request_source_sha256=REQUEST_SHA256,
        runtime_source_verified=False, runtime_config_verified=False, runtime_config=None,
        single_concurrent_batch_required=True, scheduled_concurrency_range=[2, 8],
        concurrent_draft_depth=0, conservative_multiple_unfinished_requests=True,
        clears_before_reservation=True, clears_future_concurrent_rows=True,
        preserves_original_auto_pause_and_k3_reset=True,
        true_solo_c1_policy_unchanged=True, c1_fixed_modes_unchanged=True,
        fresh_concurrent_prefix_reads_bypassed=True,
        prefix_read_bypass_latched_for_fresh_recompute=True,
        unlatched_true_solo_c1_prefix_reads_unchanged=True,
        ongoing_committed_kv_reset=False,
        underlying_shared_checkpoint_ownership_repaired=False,
        allocation_capacity_unchanged=True,
        committed_tokens_or_kv_positions_changed=False,
        numerical_or_sampling_rng_changes=False, performance_qualified=False,
        default_selected=False, promotion=False)
    state = dict(adapter_hash=adapter_hash, receipt=receipt,
                 original_install=original_install, runtime_cls=None)

    @wraps(original_budget)
    def budget(policy, ids, scheduler=None):
        depth = original_budget(policy, ids, scheduler)
        guard = scheduler is not None and _enabled(scheduler) and len(_live(scheduler)) > 1
        if 2 <= len(ids) <= 8 or (len(ids) == 1 and guard):
            if len(ids) == 1:
                request = policy.requests.get(ids[0])
                if request is not None and request.mode == 'auto':
                    request.paused = True
            return 0
        return depth

    @wraps(original_install)
    def bootstrap():
        _bind_scheduler(module, state)

    cls.budget = budget
    module.install = bootstrap
    state.update(budget_wrapper=budget, install_wrapper=bootstrap)
    cls._ornith_concurrent_k0_install = state
    return _emit_install(state)
