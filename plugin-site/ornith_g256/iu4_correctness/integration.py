"""Portable production setup around the released IU4 lifecycle.

The proposed concurrent K0 guard still needs matched runtime qualification.
Numerical and real serving qualification are separate release receipts.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PLUGIN = ROOT.parent


def verify_sources():
    pins = json.loads((ROOT / 'source-pins.json').read_text())['files']
    for name, expected in pins.items():
        actual = hashlib.sha256((PLUGIN / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('IU4 correctness source differs: ' + name)
    return pins


def load_selection(require_concurrency=False):
    value = json.loads((ROOT / 'selection.json').read_text())
    assert value['variant'] == 'all'
    assert value['repairs'] == dict(ba=True, router='all', shared=True)
    assert value['active_c1_depths'] == [0, 3, 7]
    assert value['default_policy'] == 'auto' and value['initial_auto_depth'] == 3
    assert value['auto_ceiling'] == 7 and value['allocation_depth'] == 15
    assert value['context'] == 262144 and value['max_sequences'] == 8 and value['kv_cache_gib'] == 44
    assert value['native_forward'] and value['filter_backend'] == 'native-pytorch'
    assert value['mixed_routed_adapter'] is False
    if require_concurrency and (type(value['concurrency_depth']) is not int or value['concurrency_depth'] not in (0, 7)):
        raise RuntimeError('A source-bound concurrent K0 or native K7 selection is required')
    assert value['concurrent_k0_guard'] is (value['concurrency_depth'] == 0)
    return value


def prepare_policy():
    verify_sources()
    selection = load_selection(require_concurrency=True)
    policy_name = os.environ.setdefault('ORNITH_C1_POLICY', 'auto')
    if policy_name not in ('auto', 'k0', 'k3', 'k7'):
        raise ValueError('The production IU4 policy supports auto, k0, k3 or k7; K15 is allocation capacity only')
    from . import adaptive_fallback_policy as policy
    from .adaptive_fallback_ceiling import install
    install(policy, ceiling=7)
    if selection['concurrency_depth'] == 0:
        from .adaptive_concurrent_k0 import install as install_concurrent_k0
        install_concurrent_k0(policy)
    sys.modules['ornith_g256.adaptive_c1'] = policy


def prepare_geometry():
    from .adaptive_k3_geometry import install
    install()


def validate_settings(settings):
    selection = load_selection(require_concurrency=True)
    assert settings['max_model_len'] == selection['context']
    assert settings['max_num_seqs'] == selection['max_sequences']
    assert settings['kv_cache_memory_bytes'] == selection['kv_cache_gib'] * (1 << 30)
    assert settings['async_scheduling'] is False
    assert settings['speculative_config']['num_speculative_tokens'] == selection['allocation_depth']
    assert settings['speculative_config']['num_speculative_tokens_per_batch_size'] == [(1, 1, 15), (2, 8, 7)]
    settings['worker_cls'] = 'ornith_g256.iu4_correctness.worker.Worker'
    return settings
