"""CPU policy boundaries for the real source-bound prefill service helper."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / 'plugin-site/ornith_g256'
SOURCE = PLUGIN / 'iu4_correctness/prefill_service.py'
spec = importlib.util.spec_from_file_location('prefill_service_under_test', SOURCE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def request(name, *, prompt=128000, computed=0, prefill=True, status='RUNNING', finished=False):
    return SimpleNamespace(request_id=name, num_prompt_tokens=prompt,
        num_computed_tokens=computed, is_prefill_chunk=prefill, status=status,
        is_finished=lambda: finished, spec_token_ids=[], committed_tokens=[1, 2, 3])


def scheduler(requests, running=None):
    return SimpleNamespace(requests={r.request_id: r for r in requests},
        running=requests if running is None else running,
        vllm_config=SimpleNamespace(additional_config={'ornith_g256': {'adaptive_c1_fallback': True}}),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=4480),
        max_num_scheduled_tokens=8192, prefill_capacity_bound=True,
        _is_blocked_waiting_status=lambda status: status.startswith('BLOCKED'))


class PrefillServiceTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.object(m, 'configuration', return_value=m.Configuration())
        self.config.start(); self.addCleanup(self.config.stop)
        self.out = io.StringIO()
        self.stdout = contextlib.redirect_stdout(self.out)
        self.stdout.__enter__(); self.addCleanup(self.stdout.__exit__, None, None, None)
        self.calls = []

    def original(self, s, throttle_prefills=False):
        result = dict(defer=throttle_prefills, capacity=s.prefill_capacity_bound,
            threshold=s.scheduler_config.long_prefill_token_threshold,
            budget=s.max_num_scheduled_tokens)
        self.calls.append(result)
        return result

    def mixed(self):
        return scheduler([request('decode', prompt=8000, computed=8001, prefill=False), request('long')])

    def test_first_chunk_then_fifteen_decode_turns_no_request_state_changes(self):
        s = self.mixed()
        before = [(r.num_computed_tokens, list(r.committed_tokens), list(r.spec_token_ids)) for r in s.running]
        results = [m.schedule(self.original, s) for _ in range(33)]
        self.assertEqual([i for i, r in enumerate(results) if not r['defer']], [0, 16, 32])
        self.assertTrue(all(r['threshold'] == 1120 and r['budget'] == 1121 for r in results))
        self.assertTrue(all(not r['capacity'] for r in results if r['defer']))
        self.assertEqual(s.scheduler_config.long_prefill_token_threshold, 4480)
        self.assertEqual(s.max_num_scheduled_tokens, 8192)
        self.assertTrue(s.prefill_capacity_bound)
        self.assertEqual(before, [(r.num_computed_tokens, r.committed_tokens, r.spec_token_ids) for r in s.running])
        marker = json.loads(self.out.getvalue().split('ORNITH_PREFILL_SERVICE ', 1)[1])
        self.assertEqual(marker['source_sha256'], hashlib.sha256(SOURCE.read_bytes()).hexdigest())

    def test_all_prefill_keeps_native_throughput_configuration(self):
        s = scheduler([request('one'), request('two')])
        self.assertEqual(m.schedule(self.original, s), dict(defer=False, capacity=True, threshold=4480, budget=8192))

    def test_short_prompt_boundary_is_unchanged(self):
        s = self.mixed(); s.requests['long'].num_prompt_tokens = 8192
        self.assertEqual(m.schedule(self.original, s)['threshold'], 4480)
        s.requests['long'].num_prompt_tokens = 8193
        self.assertEqual(m.schedule(self.original, s)['threshold'], 1120)

    def test_blocked_or_finished_long_request_does_not_throttle(self):
        for field, value in [('status', 'BLOCKED_REMOTE'), ('is_finished', lambda: True)]:
            s = self.mixed(); setattr(s.requests['long'], field, value)
            self.assertEqual(m.schedule(self.original, s)['budget'], 8192)

    def test_waiting_long_request_gets_same_protection(self):
        s = self.mixed(); s.running = [s.requests['decode']]
        s.requests['long'].status = 'WAITING'
        self.assertEqual(m.schedule(self.original, s)['budget'], 1121)

    def test_eight_slot_total_prefill_budget_does_not_multiply(self):
        s = scheduler([request('decode', computed=128001, prefill=False)] + [request(str(i)) for i in range(7)])
        self.assertEqual(m.schedule(self.original, s)['budget'], 1121)
        s = scheduler([request(str(i), computed=128001, prefill=False) for i in range(7)] + [request('long')])
        self.assertEqual(m.schedule(self.original, s)['budget'], 1127)

    def test_external_throttling_is_preserved(self):
        s = self.mixed()
        self.assertTrue(m.schedule(self.original, s, True)['defer'])

    def test_smaller_native_limits_are_never_expanded(self):
        s = self.mixed(); s.max_num_scheduled_tokens = 900; s.scheduler_config.long_prefill_token_threshold = 560
        result = m.schedule(self.original, s)
        self.assertEqual((result['budget'], result['threshold']), (900, 560))
        self.assertEqual((s.max_num_scheduled_tokens, s.scheduler_config.long_prefill_token_threshold), (900, 560))

    def test_exception_restores_configuration_and_capacity(self):
        s = self.mixed(); s._ornith_prefill_service_step = 1
        def fail(s, throttle_prefills=False):
            self.assertTrue(throttle_prefills); self.assertFalse(s.prefill_capacity_bound)
            raise RuntimeError('native reservation failed')
        with self.assertRaisesRegex(RuntimeError, 'native reservation failed'):
            m.schedule(fail, s)
        self.assertEqual((s.max_num_scheduled_tokens, s.scheduler_config.long_prefill_token_threshold), (8192, 4480))
        self.assertTrue(s.prefill_capacity_bound)

    def test_capacity_result_from_native_admission_is_retained(self):
        s = self.mixed()
        def update(s, throttle_prefills=False):
            s.prefill_capacity_bound = False
            return object()
        m.schedule(update, s)
        self.assertFalse(s.prefill_capacity_bound)

    def test_fully_decoding_or_sole_request_resets_service_phase(self):
        s = self.mixed(); m.schedule(self.original, s)
        s.requests['long'].num_computed_tokens = 128001
        s.requests['long'].is_prefill_chunk = False
        self.assertEqual(m.schedule(self.original, s)['budget'], 8192)
        self.assertEqual(s._ornith_prefill_service_step, 0)
        s.requests['long'].num_computed_tokens = 0; s.requests['long'].is_prefill_chunk = True
        self.assertFalse(m.schedule(self.original, s)['defer'])

    def test_disabled_control_calls_original_without_reading_runtime_state(self):
        with patch.object(m, 'configuration', return_value=m.Configuration(cadence=1)):
            s = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))
            seen = []
            result = object()
            def original(scheduler, throttle_prefills=False):
                seen.append((scheduler, throttle_prefills)); return result
            self.assertIs(m.schedule(original, s, True), result)
            self.assertEqual(seen, [(s, True)])

    def test_invalid_environment_values_are_refused(self):
        for kwargs in ({'cadence': 0}, {'cadence': 65}, {'long_threshold': -1}, {'mixed_chunk_tokens': 2240}):
            with self.assertRaises(ValueError): m.Configuration(**kwargs)


if __name__ == '__main__':
    unittest.main()
