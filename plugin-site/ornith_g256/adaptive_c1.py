"""Private reversible C1 policy pilot; fixed controls and C2–8 DF7 retained."""
from collections import deque
from functools import wraps
from pathlib import Path
import json
import os
import time

CONTROL = Path(__file__).resolve().parents[3] / 'policy-mode'
_INSTALLED = False


class RequestCost:
    """Request-local, reversible C1 probe policy. Thresholds are pilot settings."""
    MAX_CONTEXT = 32768
    INITIAL_FLOOR_MS = 19.0
    WINDOW = 8
    GAIN = .90
    MAX_PROBE_CYCLES = 4

    def __init__(self):
        self.mode = os.environ.get('ORNITH_C1_POLICY') or CONTROL.read_text().strip()
        if self.mode not in ('k0', 'k7', 'k15', 'auto'):
            raise ValueError(f'Unknown C1 mode {self.mode!r}')
        self.depth = 15 if self.mode == 'auto' else int(self.mode[1:])
        self.cycles = 0
        self.samples = deque(maxlen=self.WINDOW)
        self.floor_ms = self.INITIAL_FLOOR_MS
        self.cycle_ms = {}
        self.tokens = 0
        self.floor_tokens = 0
        self.warm = 2
        self.bad_windows = 0
        self.probe = None
        self.retry_after = {0: 0, 7: 0, 15: 0}
        self.last_15_probe = 0
        self.last_7_probe = 0
        self.paused = False
        self.long_context = False
        self.rid = None

    def event(self, kind, **values):
        print('ORNITH_C1_' + kind + ' ' + json.dumps(dict(
            request_id=self.rid, mode=self.mode, depth=self.depth,
            output_progress=self.tokens, **values)), flush=True)

    def reset_to_15(self, reason):
        old = self.depth
        self.depth = 15
        self.probe = None
        self.samples.clear()
        self.warm = 2
        self.bad_windows = 0
        self.floor_tokens = 0
        if old != 15:
            self.event('RECOVERY', previous=old, reason=reason)

    def budget(self, context):
        if self.mode != 'auto':
            return self.depth
        if context > self.MAX_CONTEXT:
            if not self.long_context:
                self.reset_to_15('context_above_32768')
                self.long_context = True
            return 15
        if self.paused:
            self.paused = False
            self.reset_to_15('return_from_concurrency')
        return self.depth

    def start_probe(self, candidate, base_cost, reason):
        incumbent = self.depth
        self.probe = dict(incumbent=incumbent, candidate=candidate,
                          baseline_ms=base_cost, samples=[], excess_ms=0.0)
        self.depth = candidate
        self.samples.clear()
        self.warm = 1
        self.bad_windows = 0
        if candidate == 15:
            self.last_15_probe = self.tokens
        if candidate == 7:
            self.last_7_probe = self.tokens
        self.event('PROBE', incumbent=incumbent, candidate=candidate,
                   baseline_ms_per_token=base_cost, reason=reason)

    def finish_probe(self, accept, cost, reason):
        probe = self.probe
        assert probe is not None
        candidate, incumbent = probe['candidate'], probe['incumbent']
        self.depth = candidate if accept else incumbent
        self.probe = None
        self.samples.clear()
        self.warm = 1
        self.floor_tokens = 0
        if not accept:
            self.retry_after[candidate] = self.tokens + 64
        if accept and incumbent == 15:
            self.last_15_probe = self.tokens
            self.retry_after[15] = self.tokens + 32
        self.event('PROBE_RESULT', incumbent=incumbent, candidate=candidate,
                   accepted=accept, cost_ms_per_token=cost,
                   reference_ms_per_token=probe['baseline_ms'],
                   observed_cycles=len(probe['samples']), excess_ms=probe['excess_ms'],
                   reason=reason)
        # Failure of DF7 must not block an independently promising floor.
        # The first paired run exposed repeated 15->7 rejections with K0 never
        # reachable, despite measured DF15 costs far above ordinary decode.
        if (not accept and candidate == 7 and incumbent == 15
                and probe['baseline_ms'] >= self.floor_ms / self.GAIN
                and self.tokens >= self.retry_after[0]):
            self.start_probe(0, probe['baseline_ms'], 'rejected_df7_try_floor_directly')
            return
        # Full seven-token proposals are direct evidence of a saturated DF7
        # block. Probe DF15 promptly; this still requires a measured gain.
        if accept and candidate == 7 and incumbent == 0:
            filled = sum(progress == 8 for _, progress in probe['samples'])
            if filled >= 2 and self.tokens >= self.retry_after[15]:
                self.start_probe(15, cost, 'recovered_full_df7_proposals')

    def observe(self, elapsed_ms, progressed, rid, actual, future, context):
        self.rid = rid
        self.tokens += progressed
        if actual == 0:
            self.floor_tokens += progressed
        self.cycles += 1
        if self.mode == 'auto':
            self.budget(context)
        steady = actual == future == self.depth
        if self.probe is not None:
            # Include transition/warm work as probe overhead, but never use it
            # as a stationary mode-cost estimate.
            self.probe['excess_ms'] += max(
                0.0, elapsed_ms - progressed * self.probe['baseline_ms'])
        if not steady:
            self.event('TRANSITION', actual_depth=actual, future_depth=future,
                       elapsed_ms=elapsed_ms, progressed=progressed)
            return
        if self.mode == 'auto' and self.long_context:
            return
        if self.warm:
            self.warm -= 1
            return
        self.samples.append((elapsed_ms, progressed))
        if self.mode != 'auto':
            if self.cycles in (2, 8, 16) or self.cycles % 32 == 0:
                ms = sum(x[0] for x in self.samples)
                progress = sum(x[1] for x in self.samples)
                self.event('WINDOW', actual_depth=actual, cycle=self.cycles,
                           elapsed_ms=ms, progressed=progress,
                           tg=1000*progress/ms if ms else None)
            return
        if self.probe is not None:
            probe = self.probe
            probe['samples'].append((elapsed_ms, progressed))
            n = len(probe['samples'])
            ms = sum(x[0] for x in probe['samples'])
            progress = sum(x[1] for x in probe['samples'])
            cost = ms / progress
            if n < 2:
                return
            if n == 2 and cost <= self.GAIN*probe['baseline_ms'] and actual == 7 and all(p == 8 for _, p in probe['samples']):
                self.cycle_ms[actual] = ms/n
                self.finish_probe(True, cost, 'two_full_df7_blocks')
            elif cost >= 1.25*probe['baseline_ms'] or (probe['excess_ms'] >= 100.0 and cost >= probe['baseline_ms']):
                self.finish_probe(False, cost, 'quick_loss_or_probe_cost_cap')
            elif n >= self.MAX_PROBE_CYCLES:
                self.cycle_ms[actual] = ms/n
                if actual == 0:
                    self.floor_ms = cost
                self.finish_probe(cost <= self.GAIN*probe['baseline_ms'], cost, 'four_steady_cycles')
            return
        # Bound recovery latency by output progress, including warm K0 work.
        if actual == 0 and self.floor_tokens >= 63:
            ms = sum(x[0] for x in self.samples)
            progress = sum(x[1] for x in self.samples)
            self.floor_ms = ms/progress
            self.floor_tokens = 0
            self.start_probe(7, self.floor_ms, 'floor_recovery_after_63_tokens')
            return
        if len(self.samples) < self.WINDOW:
            return
        samples = list(self.samples)
        self.samples.clear()
        ms = sum(x[0] for x in samples)
        progress = sum(x[1] for x in samples)
        cost = ms/progress
        self.cycle_ms[actual] = ms/len(samples)
        self.event('WINDOW', actual_depth=actual, cycle=self.cycles,
                   elapsed_ms=ms, progressed=progress, tg=1000/cost,
                   floor_ms_per_token=self.floor_ms)
        if actual == 0:
            self.floor_ms = cost
            return
        if actual == 15:
            # Prefix truncation predicts opportunity only, never a measured
            # speed claim: Q8 and Q16 noncausal proposals need not be identical.
            cycle7 = self.cycle_ms.get(7, self.cycle_ms[15]*.76)
            predicted7 = cycle7*len(samples)/sum(min(p, 8) for _, p in samples)
            promising = predicted7 <= self.GAIN*cost and cost >= .85*self.floor_ms
            self.bad_windows = self.bad_windows + 1 if promising else 0
            if self.bad_windows >= 2 and self.tokens >= self.retry_after[7]:
                self.start_probe(7, cost, 'two_windows_predict_at_least_10pct_gain')
            return
        if actual == 7:
            filled = sum(p == 8 for _, p in samples)
            if (self.tokens >= self.retry_after[15] and
                    (filled >= 2 or self.tokens-self.last_15_probe >= 64)):
                self.start_probe(15, cost, 'df7_full_blocks_or_periodic_df15_recovery')
            elif cost >= 1.05*self.floor_ms and self.tokens >= self.retry_after[0]:
                self.start_probe(0, cost, 'df7_cost_above_measured_floor')


class Policy:
    def __init__(self):
        self.requests = {}
        self.pending = None
        self.observing = None
        self.previous_mode = None

    def budget(self, ids, scheduler=None):
        if len(ids) != 1:
            if len(ids) > 1:
                for rid in ids:
                    state = self.requests.get(rid)
                    if state is not None and state.mode == 'auto':
                        state.paused = True
            return 7
        rid = ids[0]
        if rid not in self.requests:
            self.requests[rid] = RequestCost()
        state = self.requests[rid]
        state.rid = rid
        context = 0 if scheduler is None else scheduler.requests[rid].num_computed_tokens
        return state.budget(context)


def _policy(scheduler):
    if not scheduler.vllm_config.additional_config.get('ornith_g256', {}).get('adaptive_c1_fallback'):
        return None
    if not hasattr(scheduler, '_ornith_c1_cost_policy'):
        scheduler._ornith_c1_cost_policy = Policy()
    return scheduler._ornith_c1_cost_policy


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    schedule_original = Scheduler.schedule
    update_original = Scheduler.update_from_output
    stats_original = Scheduler.make_spec_decoding_stats
    draft_original = Scheduler.update_draft_token_ids

    @wraps(schedule_original)
    def schedule(self, *args, **kwargs):
        policy = _policy(self)
        if policy is None:
            return schedule_original(self, *args, **kwargs)
        policy.requests = {rid: state for rid, state in policy.requests.items()
                           if rid in self.requests and not self.requests[rid].is_finished()}
        started = time.monotonic()
        output = schedule_original(self, *args, **kwargs)
        ids = list(output.num_scheduled_tokens)
        future = policy.budget(ids, self)
        output.num_spec_tokens_to_schedule = future
        policy.pending = dict(started=started, ids=ids, future=future)
        mode = (len(ids), future)
        if ids and mode != policy.previous_mode:
            print('ORNITH_C1_MODE ' + json.dumps(dict(requests=len(ids), draft=future)), flush=True)
            policy.previous_mode = mode
        return output

    @wraps(update_original)
    def update(self, scheduler_output, model_runner_output):
        policy = _policy(self)
        if policy is None:
            return update_original(self, scheduler_output, model_runner_output)
        ids = list(scheduler_output.num_scheduled_tokens)
        rid = ids[0] if len(ids) == 1 else None
        cached = scheduler_output.scheduled_cached_reqs
        depth = len(scheduler_output.scheduled_spec_decode_tokens.get(rid, ()))
        eligible = (rid is not None and not scheduler_output.scheduled_new_reqs
                    and rid in cached.req_ids and not cached.is_context_phase(rid)
                    and depth in (0, 7, 15)
                    and scheduler_output.num_scheduled_tokens[rid] == depth + 1)
        policy.observing = dict(rid=rid, accepted=0 if depth == 0 else None) if eligible else None
        try:
            result = update_original(self, scheduler_output, model_runner_output)
            seen = policy.observing
            if seen is not None and seen['accepted'] is not None and policy.pending is not None:
                policy.pending['observation'] = (rid, seen['accepted'] + 1, depth)
            return result
        finally:
            policy.observing = None

    @wraps(stats_original)
    def stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
              num_invalid_spec_tokens, request_id):
        result = stats_original(self, spec_decoding_stats, num_draft_tokens,
                                num_accepted_tokens, num_invalid_spec_tokens, request_id)
        policy = getattr(self, '_ornith_c1_cost_policy', None)
        seen = None if policy is None else policy.observing
        if (seen is not None and request_id == seen['rid']
                and 0 <= num_accepted_tokens <= num_draft_tokens
                and not (num_invalid_spec_tokens or {}).get(request_id, 0)):
            seen['accepted'] = num_accepted_tokens
        return result

    @wraps(draft_original)
    def draft(self, draft_token_ids):
        result = draft_original(self, draft_token_ids)
        policy = _policy(self)
        if policy is None:
            return result
        pending = policy.pending
        policy.pending = None
        if pending is None:
            return result
        observation = pending.get('observation')
        if observation is not None:
            rid, progressed, depth = observation
            state = policy.requests.get(rid)
            request = self.requests.get(rid)
            if state is not None and request is not None and not request.is_finished():
                state.observe((time.monotonic()-pending['started'])*1000, progressed, rid, depth, pending['future'], request.num_computed_tokens)
        # Truncate only before reservation, retaining the already tested K0
        # context updates and homogeneous Q8 batch path on later arrivals.
        live = [rid for rid in pending['ids'] if rid in self.requests
                and not self.requests[rid].is_finished()]
        budget = policy.budget(live, self)
        if len(live) == 1:
            del self.requests[live[0]].spec_token_ids[budget:]
        return result

    Scheduler.schedule = schedule
    Scheduler.update_from_output = update
    Scheduler.make_spec_decoding_stats = stats
    Scheduler.update_draft_token_ids = draft
    _INSTALLED = True
