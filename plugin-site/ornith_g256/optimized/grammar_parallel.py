"""Parallelize independent vLLM grammar chains without changing their semantics.

Copyright 2026 Ciru. Licensed under Apache-2.0.
The original runtime implements every mask, reasoning transition and rollback.
Each worker receives a disjoint view of the original output buffer and exactly
one request, so the original speculative sequence remains ordered per matcher.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass
import functools
import hashlib
import inspect
import os
from pathlib import Path
import time

BASELINE_SHA256 = '057f1ac57ec7d5f28b47a0ad85e03ae568951d408ceff345dbcac4c19b5d6ee9'
MIN_SERIAL_COST_NS = 750_000


@dataclass
class MaskState:
    cost_per_request_ns: float = 0.0
    pool: ThreadPoolExecutor | None = None
    serial_batches: int = 0
    parallel_batches: int = 0


def _fill_one(original, manager, requests, req_id, scheduled, start, stop):
    # A shallow facade shares immutable configuration/backend and owns its
    # output-buffer view. All request-local state belongs to this one request.
    facade = copy(manager)
    facade._grammar_bitmask = manager._grammar_bitmask[start:stop]
    begin = time.perf_counter_ns()
    original(facade, requests, [req_id], scheduled)
    return time.perf_counter_ns() - begin


def _available_workers(count):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    return max(1, min(count, cpus // 2, 8))


def make_wrapper(original, grammar_type=None):
    if grammar_type is None:
        from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
        grammar_type = XgrammarGrammar
    @functools.wraps(original)
    def grammar_bitmask(self, requests, structured_output_request_ids, scheduled_spec_decode_tokens):
        manager = self
        request_ids = structured_output_request_ids
        scheduled = scheduled_spec_decode_tokens
        count = len(request_ids)
        # Keep C0/C1, diffusion and the existing large-batch optimization on
        # their original paths. This patch fills the small-batch serving gap.
        if count < 2 or count > 128 or manager.vllm_config.model_config.is_diffusion:
            return original(manager, requests, request_ids, scheduled)

        state = getattr(manager, '_ciru_mask_state', None)
        if state is None:
            state = manager._ciru_mask_state = MaskState()
        grammars = [requests[r].structured_output_request.grammar for r in request_ids]
        if not all(isinstance(g, grammar_type) for g in grammars):
            return original(manager, requests, request_ids, scheduled)
        unique = len({id(g) for g in grammars}) == count
        thinking = not manager.enable_in_reasoning and all(
            requests[r].structured_output_request.reasoning_ended is False for r in request_ids
        )
        parallel = (
            manager._grammar_bitmask is not None
            and unique
            and not thinking
            and state.cost_per_request_ns * count >= MIN_SERIAL_COST_NS
            and _available_workers(count) > 1
        )
        if not parallel:
            begin = time.perf_counter_ns()
            result = original(manager, requests, request_ids, scheduled)
            state.cost_per_request_ns = (time.perf_counter_ns() - begin) / count
            state.serial_batches += 1
            return result

        if state.pool is None:
            state.pool = ThreadPoolExecutor(max_workers=_available_workers(manager.vllm_config.scheduler_config.max_num_seqs), thread_name_prefix='ciru-grammar')
        futures = []
        offset = 0
        for req_id in request_ids:
            rows = len(scheduled.get(req_id, ())) + 1
            futures.append(state.pool.submit(_fill_one, original, manager, requests, req_id, scheduled, offset, offset + rows))
            offset += rows
        elapsed = []
        first_error = None
        # Drain every future before returning or raising: no worker can keep
        # writing into a buffer the scheduler might reuse for its next step.
        for future in futures:
            try:
                elapsed.append(future.result())
            except BaseException as error:
                if first_error is None:
                    first_error = error
        state.parallel_batches += 1
        if first_error is not None:
            raise first_error
        state.cost_per_request_ns = sum(elapsed) / count
        return manager._grammar_bitmask[:offset].numpy()
    return grammar_bitmask


def install():
    """Install once in the engine process, requiring the tested source identity."""
    from vllm.v1.structured_output import StructuredOutputManager
    if getattr(StructuredOutputManager, '_ciru_parallel_masks_installed', False):
        return
    source = Path(inspect.getsourcefile(StructuredOutputManager))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != BASELINE_SHA256:
        raise RuntimeError(f'Ciru parallel grammar requires its tested vLLM source; found {actual}')
    StructuredOutputManager.grammar_bitmask = make_wrapper(StructuredOutputManager.grammar_bitmask)
    StructuredOutputManager._ciru_parallel_masks_installed = True
    from vllm.logger import init_logger
    init_logger(__name__).info('Ciru parallel grammar masks R01 enabled; strict constraints and original per-request transitions retained')
