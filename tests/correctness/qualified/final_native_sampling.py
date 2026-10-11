"""Bind the unchanged native sampler before compilation, including unseeded calls.

This follows final_sampler.install. Native random_sample and its per-request
generators stay intact. Unseeded ROCm calls switch from AITER's global RNG path
to native exponential noise; natural output/acceptance/speed need fresh gates.
DFlash candidate sampling and the outer sampler's greedy fast path are intact.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import types

SOURCE_SHA256 = {
    'topk_topp_sampler.py': '60d816d829797436240bada454dd66d6ab583522656e5db5d6ed3342c109eefc',
    'topk_topp_triton.py': '255619f699898ddaaac959d9bef4494294c76d730abb8e90549afac83a9bcbe9',
    'rejection_sampler.py': '4bb87d7984d967be9c7e5b6a6b489c02571ef2f2913211cb3f1eea74b2e1f32f',
}
_SAMPLER = 'vllm.v1.sample.ops.topk_topp_sampler'
_MARKER = '_ciru_final_native_sampling_state'
_LINKS = ('sampler', '_sampler', 'topk_topp_sampler', 'samplers', 'drafter',
          'proposer', '_proposer', 'rejection_sampler', '_modules')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _function(value, name, path):
    if (not isinstance(value, types.FunctionType) or value.__name__ != name
            or value.__module__ != _SAMPLER
            or Path(value.__code__.co_filename).resolve() != path):
        raise RuntimeError('Unreviewed native sampling function: '+name)
    return value


def _describe(method):
    function = method.__func__ if isinstance(method, types.MethodType) else method
    return {'module': function.__module__, 'qualname': function.__qualname__,
            'source_path': str(Path(function.__code__.co_filename).resolve()),
            'first_line': function.__code__.co_firstlineno}


def _instances(runner, cls):
    """Walk sampler/proposer links only; never traverse weights or tensors."""
    result = {}

    def visit(value, path, ancestors=(), depth=0):
        if value is None or depth > 8 or id(value) in ancestors:
            return
        if isinstance(value, cls):
            if type(value) is not cls:
                raise RuntimeError('Unreviewed TopKTopPSampler subclass at '+path)
            result.setdefault(id(value), [value, []])[1].append(path)
            return
        ancestors = ancestors + (id(value),)
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, (str, int)):
                    if (path.endswith('._modules') and key not in _LINKS
                            and not str(key).isdigit() and not isinstance(child, cls)):
                        continue
                    visit(child, path+'['+repr(key)+']', ancestors, depth+1)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path+'['+str(index)+']', ancestors, depth+1)
        else:
            fields = vars(value) if hasattr(value, '__dict__') else {}
            for key in _LINKS:
                if key in fields:
                    visit(fields[key], path+'.'+key, ancestors, depth+1)

    visit(runner, 'worker.model_runner')
    return sorted(result.values(), key=lambda item: min(item[1]))


def _validate_forward(instance, allowed):
    forward = instance.forward
    if (not isinstance(forward, types.MethodType) or forward.__self__ is not instance
            or forward.__func__ not in allowed):
        raise RuntimeError('Unreviewed instance sampler.forward binding')
    return forward


def install(worker, evidence_dir=None):
    """Source-bound existing/future TopKTopPSampler binding; install pre-compile."""
    sampler = importlib.import_module(_SAMPLER)
    modules = [sampler,
               importlib.import_module('vllm.v1.sample.ops.topk_topp_triton'),
               importlib.import_module('vllm.v1.sample.rejection_sampler')]
    paths = {Path(m.__file__).name: Path(m.__file__).resolve() for m in modules}
    hashes = {name: _digest(path) for name, path in paths.items()}
    if hashes != SOURCE_SHA256:
        raise RuntimeError('Native sampling source hashes changed: '+json.dumps(hashes))
    source = paths['topk_topp_sampler.py']
    native_filter = _function(sampler.apply_top_k_top_p_pytorch,
                              'apply_top_k_top_p_pytorch', source)
    filter_state = getattr(sampler, '_ciru_final_sampler_state', None)
    if (not isinstance(filter_state, dict) or filter_state.get('native') is not native_filter
            or sampler.apply_top_k_top_p is not native_filter
            or modules[2].apply_top_k_top_p is not native_filter):
        raise RuntimeError('Install final_sampler before final_native_sampling')
    random_sample = _function(sampler.random_sample, 'random_sample', source)
    cls = sampler.TopKTopPSampler
    if cls.__module__ != _SAMPLER or cls.__name__ != 'TopKTopPSampler':
        raise RuntimeError('Native sampler class changed')
    native = _function(cls.forward_native, 'forward_native', source)
    allowed = {native}
    for name in ('forward_hip', 'forward_cuda', 'forward_cpu', 'forward_xpu'):
        function = getattr(cls, name, None)
        if function is not None:
            allowed.add(_function(function, name, source))
    state = getattr(sampler, _MARKER, None)
    if state is None:
        original_init = _function(cls.__init__, '__init__', source)
    else:
        if (cls is not state['class'] or cls.__init__ is not state['constructor']
                or native is not state['native'] or random_sample is not state['random_sample']):
            raise RuntimeError('Installed native sampler class/functions were rebound')
        original_init = state['original_init']
    runner = worker.model_runner
    target = runner.sampler.topk_topp_sampler
    if type(target) is not cls:
        raise RuntimeError('Required target sampler is not the reviewed TopKTopPSampler')
    instances = _instances(runner, cls)
    if not any(instance is target for instance, _ in instances):
        raise RuntimeError('Required target sampler binding was not discovered')
    before = {}
    records = []
    for instance, aliases in instances:
        forward = _validate_forward(instance, allowed if state is None else {native})
        before[id(instance)] = forward
        old = state['before'].get(id(instance), forward) if state else forward
        records.append({'paths': sorted(set(aliases)), 'before': _describe(old),
                        'after': _describe(instance.forward_native)})

    def constructor(instance, *args, **kwargs):
        if cls.forward_native is not native or sampler.random_sample is not random_sample:
            raise RuntimeError('Native sampler functions changed before future construction')
        original_init(instance, *args, **kwargs)
        old = _validate_forward(instance, allowed)
        instance.forward = types.MethodType(native, instance)
        print('ORNITH_FINAL_NATIVE_SAMPLING_INSTANCE '+json.dumps({
            'schema': 'ciru.ornith.final_native_sampling.instance.v1', 'pid': os.getpid(),
            'before': _describe(old), 'after': _describe(instance.forward)}, sort_keys=True), flush=True)

    constructor_to_install = state['constructor'] if state else constructor
    receipt = {
        'schema': 'ciru.ornith.final_native_sampling.binding.v1', 'pid': os.getpid(),
        'adapter_sha256': _digest(__file__),
        'runtime_source_sha256': hashes,
        'runtime_source_paths': {name: str(path) for name, path in paths.items()},
        'class': _SAMPLER+'.TopKTopPSampler',
        'class_constructor_before': _describe(original_init),
        'future_constructor_binds_native_after_original_init': True,
        'canonical_forward': _describe(native), 'native_random_sample': _describe(random_sample),
        'instances': records, 'existing_unique_instances': len(instances),
        'filter_and_native_forward_source_unchanged': True,
        'sampling_parameters_and_seeded_generators_unchanged': True,
        'unseeded_backend_rng_consumption_changed': any(r['before']['qualname'] != native.__qualname__
                                                      for r in records),
        'unseeded_rng_backend': 'native random_sample global exponential-noise path',
        'outer_sampler_greedy_branch_unchanged': True,
        'dflash_candidate_sampling_and_greedy_unchanged': True,
        'install_before_compile_and_oracle_hooks': True,
        'full_pq_or_serving_qualified': False, 'promotion': False,
    }
    output = None if evidence_dir is None else Path(evidence_dir)/'final-native-sampling-binding.json'
    if output is not None and output.exists():
        if state is None or json.loads(output.read_text()) != receipt:
            raise RuntimeError('Refusing to replace conflicting native sampling receipt')
    for instance, _ in instances:
        instance.forward = types.MethodType(native, instance)
    cls.__init__ = constructor_to_install
    originals = dict(state['before']) if state else {}
    originals.update({key: value for key, value in before.items() if key not in originals})
    setattr(sampler, _MARKER, {'class': cls, 'constructor': constructor_to_install,
                             'original_init': original_init, 'native': native,
                             'random_sample': random_sample, 'before': originals, 'receipt': receipt})
    try:
        if output is not None and not output.exists():
            with output.open('x') as stream:
                stream.write(json.dumps(receipt, indent=2)+'\n')
    except Exception:
        for instance, _ in instances:
            instance.forward = before[id(instance)]
        cls.__init__ = state['constructor'] if state else original_init
        if state is None:
            delattr(sampler, _MARKER)
        else:
            setattr(sampler, _MARKER, state)
        raise
    print('ORNITH_FINAL_NATIVE_SAMPLING '+json.dumps(receipt, sort_keys=True), flush=True)
    return receipt
