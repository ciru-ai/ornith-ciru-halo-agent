"""Optional source-bound AUTO ceiling, preserving the frozen policy modules.

Install before native package import/worker setup. Allocation and explicit
fixed-depth diagnostics are unchanged; only above-ceiling AUTO probes skip.
"""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path


POLICY_SOURCE_SHA256 = frozenset({
    '1d5284ab49e879df8e2e0c6af374edd37555d03199032589adad46aafd01c81c',
    '0cb7ffc60d801b6aafa0ef1a6db5b1dea7f6249c3c9a563625d54a4cf1c2f011',
    '6e8be45ea6cee7b995009f967f08190a32b18327a42fd768b252faa875dbed5e',
})


def install(module, ceiling=7):
    """Restrict AUTO probes on one exact reviewed slow/fast RequestCost class."""
    if type(ceiling) is not int or ceiling not in (7, 15):
        raise ValueError('AUTO ceiling must be integer 7 or 15')
    policy_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if policy_hash not in POLICY_SOURCE_SHA256:
        raise RuntimeError('AUTO ceiling policy source differs from frozen slow/fast')
    adapter_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cls = module.RequestCost
    existing = cls.__dict__.get('_ornith_auto_ceiling_install')
    if existing is not None:
        if existing['ceiling'] != ceiling:
            raise RuntimeError('AUTO ceiling conflicting reinstall')
        if (cls.start_probe is not existing['wrapper']
                or existing['policy_hash'] != policy_hash
                or existing['adapter_hash'] != adapter_hash):
            raise RuntimeError('AUTO ceiling installed wrapper/source changed')
        receipt = dict(existing['receipt'], pid=os.getpid())
        # A fork may retain an installed class. Bind its inherited wrapper to
        # this engine PID without installing a second interposer.
        if receipt['pid'] != existing['receipt']['pid']:
            print('ORNITH_AUTO_CEILING_INSTALLED ' + json.dumps(receipt), flush=True)
            existing['receipt'] = receipt
        return receipt

    original = cls.start_probe
    if original.__module__ != module.__name__:
        raise RuntimeError('AUTO ceiling policy start_probe source is not original')

    @wraps(original)
    def start_probe(state, candidate, base_cost, reason):
        if state.mode == 'auto' and candidate > ceiling:
            state.event('PROBE_SKIPPED', requested_candidate=candidate,
                        ceiling=ceiling, reason='auto_ceiling',
                        requested_reason=reason)
            # Disabling the original upward WINDOW branch must not suppress
            # the lower-depth elseif that would otherwise become eligible.
            # Reuse its exact cost/periodic predicate and existing probe API.
            if (candidate == 15 and state.depth == 7 and state.probe is None
                    and state.tokens >= state.retry_after[3]
                    and (state.tokens - state.last_3_probe >= 64
                         or base_cost >= 1.05 * state.floor_ms)):
                return original(state, 3, base_cost,
                                'auto_ceiling_lower_depth_recovery')
            return None
        return original(state, candidate, base_cost, reason)

    receipt = dict(pid=os.getpid(), adapter_sha256=adapter_hash,
                   policy_module=module.__name__,
                   policy_source_sha256=policy_hash, ceiling=ceiling,
                   auto_only=True, allocation_depth_unchanged=15,
                   concurrency_depth_unchanged=7, numerical_changes=False,
                   preserves_lower_depth_recovery=True)
    cls.start_probe = start_probe
    cls._ornith_auto_ceiling_install = dict(ceiling=ceiling, wrapper=start_probe,
        policy_hash=policy_hash, adapter_hash=adapter_hash, receipt=receipt)
    print('ORNITH_AUTO_CEILING_INSTALLED ' + json.dumps(receipt), flush=True)
    return dict(receipt)
