"""CPU reservation-boundary checks for the optional concurrency contingency."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
STAGE = ROOT.parents[1]
SOURCES = ROOT / 'runtime-sources'


class Request:
    def __init__(self, rid, drafts=(), *, computed=3, committed=(10, 11, 12, 13), new=False):
        self.request_id = rid
        self._all_token_ids = list(committed)
        self.spec_token_ids = list(drafts)
        self.num_computed_tokens = computed
        self.num_in_flight_tokens = 0
        self.num_output_placeholders = 0
        self.finished = False
        self.new = new
        self.is_prefill_chunk = False
        self.skip_reading_prefix_cache = False

    @property
    def num_tokens_with_spec(self):
        return len(self._all_token_ids) + len(self.spec_token_ids)

    def is_finished(self):
        return self.finished


def policy_module():
    path = STAGE / 'plugin-site/ornith_g256/iu4_correctness/adaptive_fallback_policy.py'
    spec = importlib.util.spec_from_file_location('concurrent_test_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scheduler_modules():
    """Fixture executes the pinned token-reservation and new-row padding rules."""
    module = ModuleType('vllm.v1.core.sched.scheduler')
    module.__file__ = str(SOURCES / 'scheduler.py')
    module.SimpleNamespace = SimpleNamespace
    source = '''
class Scheduler:
    def __init__(self, requests):
        self.requests = {r.request_id: r for r in requests}
        self.max_num_running_reqs = 8
        self.num_spec_tokens = 15
        self.num_lookahead_tokens = 15
        self.use_pp = self.use_v2_model_runner = False
        self.scheduler_config = SimpleNamespace(async_scheduling=False)
        self.vllm_config = SimpleNamespace(max_concurrent_batches=1,
            additional_config={'ornith_g256': {'adaptive_c1_fallback': True}})
        self.admit_ids = list(self.requests)
        self.calls = []
        self.connector = None
        self.kv_cache_manager = SimpleNamespace(empty_kv_cache_blocks=())
    def _get_local_prefix_cache_hit(self, request):
        self.calls.append(('native_prefix_lookup', request.request_id))
        return (('cached-block',), 2, 1120, False)
    def schedule(self, *args, **kwargs):
        if getattr(self, 'raise_during_schedule', False):
            raise ValueError('parent reservation failure')
        counts, scheduled_drafts, before = {}, {}, {}
        for rid in self.admit_ids:
            r = self.requests[rid]
            if r.is_finished():
                continue
            if r.num_computed_tokens == 0:
                _, r.num_computed_tokens, r.shared_prefix_boundary, _ = self._get_local_prefix_cache_hit(r)
            before[rid] = r.num_computed_tokens
            count = r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens
            drafts = list(r.spec_token_ids)
            if r.new and count == 1 and counts and self.num_spec_tokens > 0:
                count = 1 + self.num_spec_tokens
                drafts = [-1] * self.num_spec_tokens
            counts[rid] = count
            if drafts:
                scheduled_drafts[rid] = drafts
            r.spec_token_ids = []
            r.num_computed_tokens += count
            r.num_in_flight_tokens += count
        self.calls.append(('reserve', self.num_spec_tokens, before, dict(counts)))
        return SimpleNamespace(num_scheduled_tokens=counts, total_num_scheduled_tokens=sum(counts.values()),
            scheduled_spec_decode_tokens=scheduled_drafts, num_spec_tokens_to_schedule=self.num_spec_tokens)
    def update_from_output(self, *args, **kwargs):
        return 'parent-update'
    def make_spec_decoding_stats(self, *args, **kwargs):
        return 'parent-stats'
    def update_draft_token_ids(self, values):
        self.calls.append(('draft',))
        for rid, drafts in zip(values.req_ids, values.draft_token_ids):
            if rid in self.requests and not self.requests[rid].is_finished():
                self.requests[rid].spec_token_ids = list(drafts)
        return 'parent-draft'
'''
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    request = ModuleType('vllm.v1.request')
    request.__file__ = str(SOURCES / 'request.py')
    return module, request


class ConcurrentK0Tests(unittest.TestCase):
    def setUp(self):
        self.module = policy_module()
        if os.environ.get('ORNITH_CONCURRENT_K0_RED_CONTROL') == '1':
            self.install = lambda module: {}
            self.adapter = None
        else:
            path = STAGE / 'plugin-site/ornith_g256/iu4_correctness/adaptive_concurrent_k0.py'
            package = ModuleType('staged_iu4')
            package.__path__ = [str(path.parent)]
            sys.modules['staged_iu4'] = package
            env = patch.dict(os.environ, CIRU_PREFILL_CADENCE='1')
            env.start(); self.addCleanup(env.stop)
            spec = importlib.util.spec_from_file_location('staged_iu4.adaptive_concurrent_k0', path)
            adaptive_concurrent_k0 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adaptive_concurrent_k0)
            self.adapter = adaptive_concurrent_k0
            self.install = self.adapter.install
        self.native, self.request_module = scheduler_modules()
        self.modules = patch.dict(sys.modules, {
            'vllm.v1.core.sched.scheduler': self.native,
            'vllm.v1.request': self.request_module,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.output = io.StringIO()
        with contextlib.redirect_stdout(self.output):
            self.receipt = self.install(self.module)
            self.module.install()

    def schedule(self, scheduler):
        with patch.dict(os.environ, ORNITH_C1_POLICY='auto'), contextlib.redirect_stdout(self.output):
            return scheduler.schedule()

    def test_first_c1_to_c2_transition_has_zero_reserved_drafts(self):
        a = Request('a', [20, 21, 22])
        b = Request('b', [30] * 7)
        scheduler = self.native.Scheduler([a, b])
        committed = {r.request_id: list(r._all_token_ids) for r in (a, b)}
        result = self.schedule(scheduler)
        self.assertEqual(result.scheduled_spec_decode_tokens, {})
        self.assertEqual(result.num_scheduled_tokens, {'a': 1, 'b': 1})
        self.assertEqual(result.num_spec_tokens_to_schedule, 0)
        self.assertEqual(scheduler.num_spec_tokens, 15)
        self.assertEqual(scheduler.num_lookahead_tokens, 15)
        self.assertEqual({r.request_id: r._all_token_ids for r in (a, b)}, committed)
        self.assertEqual([a.num_computed_tokens, b.num_computed_tokens], [4, 4])
        self.assertEqual([a.num_in_flight_tokens, b.num_in_flight_tokens], [1, 1])

    def test_new_resume_padding_cannot_reintroduce_speculative_rows(self):
        scheduler = self.native.Scheduler([Request('a'), Request('b', new=True)])
        result = self.schedule(scheduler)
        self.assertEqual(result.num_scheduled_tokens, {'a': 1, 'b': 1})
        self.assertEqual(result.scheduled_spec_decode_tokens, {})
        self.assertEqual(scheduler.calls[0][1], 0)

    def test_fresh_concurrent_admission_bypasses_prefix_read_before_reservation(self):
        a, b = Request('a', computed=0), Request('b', computed=0)
        scheduler = self.native.Scheduler([a, b])
        committed = copy.deepcopy([a._all_token_ids, b._all_token_ids])
        result = self.schedule(scheduler)
        self.assertEqual(result.num_scheduled_tokens, {'a': 4, 'b': 4})
        self.assertFalse(any(call[0] == 'native_prefix_lookup' for call in scheduler.calls))
        self.assertEqual([a._all_token_ids, b._all_token_ids], committed)
        self.assertEqual(scheduler.num_spec_tokens, 15)
        self.assertEqual(scheduler.num_lookahead_tokens, 15)
        for request in (a, b):
            self.assertTrue(request.skip_reading_prefix_cache)
            self.assertTrue(request._ornith_concurrent_prefix_read_isolated)
            self.assertEqual(request.shared_prefix_boundary, 0)
        markers = [json.loads(line.split(' ', 1)[1]) for line in self.output.getvalue().splitlines()
                   if line.startswith('ORNITH_CONCURRENCY_PREFIX_READ_BYPASS ')]
        self.assertEqual({marker['request_id'] for marker in markers}, {'a', 'b'})
        for marker in markers:
            self.assertEqual(marker['num_computed_tokens'], 0)
            self.assertTrue(marker['fresh_only'])
            self.assertTrue(marker['skip_reading_prefix_cache'])
            self.assertTrue(marker['prefix_read_bypassed'])
            self.assertTrue(marker['empty_computed_blocks'])
            self.assertEqual(marker['num_local_computed_tokens'], 0)
            self.assertEqual(marker['shared_prefix_boundary'], 0)
            self.assertEqual(marker['adapter_sha256'], self.receipt['adapter_sha256'])
            self.assertEqual(marker['scheduler_source_sha256'], self.adapter.SCHEDULER_SHA256)
            self.assertEqual(marker['pid'], os.getpid())

    def test_prefix_read_isolation_latches_for_fresh_recompute_after_concurrency(self):
        a, b = Request('a', computed=0), Request('b', computed=0)
        scheduler = self.native.Scheduler([a, b])
        with contextlib.redirect_stdout(self.output):
            first = scheduler._get_local_prefix_cache_hit(a)
        self.assertIs(first[0], scheduler.kv_cache_manager.empty_kv_cache_blocks)
        self.assertEqual(first[1:], (0, 0, False))
        b.finished = True
        with contextlib.redirect_stdout(self.output):
            later = scheduler._get_local_prefix_cache_hit(a)
        self.assertIs(later[0], first[0])
        self.assertEqual(later[1:], (0, 0, False))
        self.assertEqual(scheduler.calls, [])

    def test_prefix_read_isolation_preserves_solo_and_ongoing_native_context(self):
        for enabled, computed, second in [(True, 0, False), (True, 3, True), (False, 0, True)]:
            a = Request('a', computed=computed)
            scheduler = self.native.Scheduler([a] + ([Request('b')] if second else []))
            if not enabled:
                scheduler.vllm_config.additional_config = {}
            before = copy.deepcopy(a.__dict__)
            result = scheduler._get_local_prefix_cache_hit(a)
            self.assertEqual(result, (('cached-block',), 2, 1120, False))
            self.assertEqual(a.__dict__, before)
            self.assertEqual(scheduler.calls, [('native_prefix_lookup', 'a')])

    def test_prefix_read_isolation_rejects_connector_without_mutating_request(self):
        a, b = Request('a', computed=0), Request('b', computed=0)
        scheduler = self.native.Scheduler([a, b])
        scheduler.connector = object()
        before = copy.deepcopy(a.__dict__)
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(self.output):
            scheduler._get_local_prefix_cache_hit(a)
        self.assertEqual(a.__dict__, before)
        self.assertEqual(scheduler.calls, [])

    def test_connector_rejects_fresh_schedule_before_clearing_queued_drafts(self):
        a, b = Request('a', [20], computed=0), Request('b', [30], computed=0)
        scheduler = self.native.Scheduler([a, b])
        scheduler.connector = object()
        before = copy.deepcopy([a.__dict__, b.__dict__])
        with self.assertRaises(RuntimeError):
            self.schedule(scheduler)
        self.assertEqual([a.__dict__, b.__dict__], before)
        self.assertEqual(scheduler.calls, [])

    def test_latched_ongoing_context_still_uses_native_lookup_without_reset(self):
        a = Request('a', computed=3)
        a._ornith_concurrent_prefix_read_isolated = True
        a.skip_reading_prefix_cache = True
        scheduler = self.native.Scheduler([a, Request('b')])
        before = copy.deepcopy(a.__dict__)
        result = scheduler._get_local_prefix_cache_hit(a)
        self.assertEqual(result, (('cached-block',), 2, 1120, False))
        self.assertEqual(a.__dict__, before)

    def test_prefix_read_original_must_be_the_pinned_native_method(self):
        native, request = scheduler_modules()
        native.Scheduler._get_local_prefix_cache_hit = lambda scheduler, row: None
        fresh = policy_module()
        with patch.dict(sys.modules, {'vllm.v1.core.sched.scheduler': native,
                'vllm.v1.request': request}), contextlib.redirect_stdout(self.output):
            self.install(fresh)
            with self.assertRaisesRegex(RuntimeError, 'native prefix lookup is already interposed'):
                fresh.install()

    def test_prefix_read_isolation_rejects_changed_wrapper_on_bootstrap(self):
        self.native.Scheduler._get_local_prefix_cache_hit = lambda scheduler, request: None
        with self.assertRaises(RuntimeError):
            self.module.install()

    def test_every_supported_concurrency_and_boundary_budget(self):
        for count in range(2, 9):
            requests = [Request(str(i), [90 + i] * 7) for i in range(count)]
            result = self.schedule(self.native.Scheduler(requests))
            self.assertEqual(result.total_num_scheduled_tokens, count)
            self.assertEqual(result.scheduled_spec_decode_tokens, {})
            self.assertEqual(result.num_spec_tokens_to_schedule, 0)
        policy = self.module.Policy()
        self.assertEqual(policy.budget([]), 7)
        self.assertEqual(policy.budget(list(map(str, range(9)))), 7)

    def test_parent_failure_restores_capacity_without_touching_committed_state(self):
        a, b = Request('a', [20]), Request('b', [30])
        scheduler = self.native.Scheduler([a, b])
        scheduler.raise_during_schedule = True
        before = [(list(r._all_token_ids), r.num_computed_tokens, r.num_in_flight_tokens) for r in (a, b)]
        with self.assertRaises(ValueError):
            self.schedule(scheduler)
        self.assertEqual(scheduler.num_spec_tokens, 15)
        self.assertEqual([(r._all_token_ids, r.num_computed_tokens, r.num_in_flight_tokens) for r in (a, b)], before)

    def test_future_concurrent_rows_are_cleared_after_parent_draft_update(self):
        scheduler = self.native.Scheduler([Request('a'), Request('b')])
        self.schedule(scheduler)
        values = SimpleNamespace(req_ids=['a', 'b'], draft_token_ids=[[20, 21], [30, 31]])
        with contextlib.redirect_stdout(self.output):
            result = scheduler.update_draft_token_ids(values)
        self.assertEqual(result, 'parent-draft')
        self.assertEqual(scheduler.calls[-1], ('draft',))
        self.assertEqual([r.spec_token_ids for r in scheduler.requests.values()], [[], []])

    def test_true_solo_c1_and_fixed_c1_modes_retain_native_budget(self):
        for mode, expected in [('auto', 3), ('k0', 0), ('k3', 3), ('k7', 7), ('k15', 15)]:
            scheduler = self.native.Scheduler([Request('a', [20, 21, 22])])
            with patch.dict(os.environ, ORNITH_C1_POLICY=mode), contextlib.redirect_stdout(self.output):
                result = scheduler.schedule()
            self.assertEqual(result.num_scheduled_tokens, {'a': 4})
            self.assertEqual(result.scheduled_spec_decode_tokens, {'a': [20, 21, 22]})
            self.assertEqual(result.num_spec_tokens_to_schedule, expected)

    def test_paused_auto_returns_to_original_k3_reset_and_warm_latch(self):
        scheduler = self.native.Scheduler([Request('a', [20] * 7), Request('b')])
        with patch.dict(os.environ, ORNITH_C1_POLICY='auto'):
            policy = self.module.Policy()
            self.assertEqual(policy.budget(['a'], scheduler), 0)
        state = policy.requests['a']
        state.depth = 7
        state.probe = {'incumbent': 3}
        state.samples.append((40., 8))
        scheduler._ornith_c1_cost_policy = policy
        self.schedule(scheduler)
        self.assertTrue(state.paused)
        scheduler.requests['b'].finished = True
        with contextlib.redirect_stdout(self.output):
            depth = policy.budget(['a'], scheduler)
        self.assertEqual(depth, 3)
        self.assertFalse(state.paused)
        self.assertTrue(state.startup_warm_pending)
        self.assertEqual(state.warm, 2)
        self.assertEqual(list(state.samples), [])
        self.assertIsNone(state.probe)

    def test_waiting_second_request_conservatively_guards_c1_before_reservation(self):
        scheduler = self.native.Scheduler([Request('a', [20] * 7), Request('blocked')])
        scheduler.admit_ids = ['a']
        result = self.schedule(scheduler)
        self.assertEqual(result.num_scheduled_tokens, {'a': 1})
        self.assertEqual(result.scheduled_spec_decode_tokens, {})
        self.assertEqual(result.num_spec_tokens_to_schedule, 0)
        self.assertIn('multiple_unfinished_requests', self.output.getvalue())

    def test_disabled_native_policy_and_empty_schedule_remain_native(self):
        scheduler = self.native.Scheduler([Request('a', [20]), Request('b', [30])])
        scheduler.vllm_config.additional_config = {}
        result = self.schedule(scheduler)
        self.assertEqual(result.scheduled_spec_decode_tokens, {'a': [20], 'b': [30]})
        self.assertEqual(result.num_spec_tokens_to_schedule, 15)
        scheduler = self.native.Scheduler([])
        result = self.schedule(scheduler)
        self.assertEqual(result.num_scheduled_tokens, {})
        self.assertEqual(result.num_spec_tokens_to_schedule, 7)

    def test_async_or_inflight_guard_rejects_before_clearing_any_request(self):
        for attribute, value in [('async', True), ('pp', True), ('runner', True), ('max', 9), ('inflight', 1), ('placeholder', 1)]:
            a, b = Request('a', [20]), Request('b', [30])
            scheduler = self.native.Scheduler([a, b])
            if attribute == 'async': scheduler.scheduler_config.async_scheduling = value
            if attribute == 'pp': scheduler.use_pp = value
            if attribute == 'runner': scheduler.use_v2_model_runner = value
            if attribute == 'max': scheduler.max_num_running_reqs = value
            if attribute == 'inflight': a.num_in_flight_tokens = value
            if attribute == 'placeholder': a.num_output_placeholders = value
            before = copy.deepcopy([a.__dict__, b.__dict__])
            with self.assertRaises(RuntimeError):
                self.schedule(scheduler)
            self.assertEqual([a.__dict__, b.__dict__], before)
            self.assertEqual(scheduler.calls, [])

    def test_overlapping_or_unknown_batch_queue_rejects_before_clearing(self):
        for concurrent_batches in (2, None):
            a, b = Request('a', [20]), Request('b', [30])
            scheduler = self.native.Scheduler([a, b])
            if concurrent_batches is None:
                del scheduler.vllm_config.max_concurrent_batches
            else:
                scheduler.vllm_config.max_concurrent_batches = concurrent_batches
            before = copy.deepcopy([a.__dict__, b.__dict__])
            with self.assertRaises(RuntimeError):
                self.schedule(scheduler)
            self.assertEqual([a.__dict__, b.__dict__], before)
            self.assertEqual(scheduler.calls, [])

    def test_source_bound_install_and_actual_budget_receipts(self):
        scheduler = self.native.Scheduler([Request('a', [20]), Request('b', [30])])
        self.schedule(scheduler)
        lines = self.output.getvalue().splitlines()
        installs = [json.loads(line.split(' ', 1)[1]) for line in lines if line.startswith('ORNITH_CONCURRENCY_K0_INSTALLED ')]
        self.assertTrue(installs, 'Missing source-bound concurrency installer receipt')
        receipt = installs[-1]
        self.assertTrue(receipt['conservative_multiple_unfinished_requests'])
        self.assertTrue(receipt['clears_before_reservation'])
        self.assertFalse(receipt['committed_tokens_or_kv_positions_changed'])
        self.assertIn('runtime_config_verified', installs[0])
        self.assertFalse(installs[0]['runtime_config_verified'])
        self.assertIsNone(installs[0]['runtime_config'])
        self.assertTrue(receipt['runtime_config_verified'])
        self.assertEqual(receipt['runtime_config'], dict(max_concurrent_batches=1,
            async_scheduling=False, pipeline_parallel=False, use_v2_model_runner=False,
            max_num_running_reqs=8, unfinished_requests_zero_in_flight_tokens=True,
            unfinished_requests_zero_output_placeholders=True))
        actual = [json.loads(line.split(' ', 1)[1]) for line in lines if line.startswith('ORNITH_CONCURRENCY_K0_SCHEDULE ')][-1]
        self.assertEqual(actual['scheduled_requests'], 2)
        self.assertEqual(actual['scheduled_draft_tokens'], 0)
        self.assertEqual(actual['future_draft_depth'], 0)
        self.assertEqual(actual['current_step_k0'], True)

    def test_idempotence_source_and_installed_wrapper_tamper_guards(self):
        if self.adapter is None:
            self.fail('No source-bound installer on unchanged red control')
        with contextlib.redirect_stdout(self.output):
            again = self.install(self.module)
        self.assertEqual(again['adapter_sha256'], self.receipt['adapter_sha256'])
        self.module.Policy.budget = lambda self, ids, scheduler=None: 7
        with self.assertRaises(RuntimeError):
            self.install(self.module)
        other = policy_module()
        other.__file__ = str(Path(__file__))
        with self.assertRaises(RuntimeError):
            self.install(other)

    def test_runtime_source_and_bootstrap_wrapper_tamper_guards(self):
        if self.adapter is None:
            self.fail('No runtime source guards on unchanged red control')
        self.native.__file__ = __file__
        with self.assertRaises(RuntimeError):
            self.module.install()
        self.native.__file__ = str(SOURCES / 'scheduler.py')
        self.native.Scheduler.schedule = lambda scheduler: None
        with self.assertRaises(RuntimeError):
            self.module.install()

    def test_scheduler_pin_matches_the_actual_single_rank_runtime(self):
        if self.adapter is None:
            self.fail('No actual runtime source pin on unchanged red control')
        self.assertEqual(self.adapter.SCHEDULER_SHA256,
            '30160d4df0366e22f9ff583d991e2898b3b65c5d25e471a209578336e24001b5')

    def test_child_pid_receipt_binds_the_actual_scheduler_process(self):
        if self.adapter is None:
            self.fail('No process binding on unchanged red control')
        scheduler = self.native.Scheduler([Request('a'), Request('b')])
        with patch.object(self.adapter.os, 'getpid', return_value=99999):
            self.schedule(scheduler)
        receipts = [json.loads(line.split(' ', 1)[1]) for line in self.output.getvalue().splitlines()
                    if line.startswith('ORNITH_CONCURRENCY_K0_INSTALLED ')]
        self.assertEqual(receipts[-1]['pid'], 99999)
        self.assertTrue(receipts[-1]['runtime_source_verified'])

    def test_child_runtime_configuration_proof_requires_its_own_guard_check(self):
        if self.adapter is None:
            self.fail('No process-specific runtime guard on unchanged red control')
        scheduler = self.native.Scheduler([Request('a'), Request('b')])
        self.schedule(scheduler)
        for request in scheduler.requests.values():
            request.num_in_flight_tokens = 0
            request._all_token_ids.append(14)
        with patch.object(self.adapter.os, 'getpid', return_value=99999):
            self.schedule(scheduler)
        receipts = [json.loads(line.split(' ', 1)[1]) for line in self.output.getvalue().splitlines()
                    if line.startswith('ORNITH_CONCURRENCY_K0_INSTALLED ')]
        child = [receipt for receipt in receipts if receipt['pid'] == 99999]
        self.assertEqual([receipt['runtime_config_verified'] for receipt in child], [False, True])
        self.assertIsNone(child[0]['runtime_config'])
        self.assertEqual(child[-1]['runtime_config']['max_concurrent_batches'], 1)


if __name__ == '__main__':
    unittest.main()
