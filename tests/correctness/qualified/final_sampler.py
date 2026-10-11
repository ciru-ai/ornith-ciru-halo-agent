"""Use the pinned native PyTorch filter for target sampling and verification.

Install before compilation and before the oracle captures its local filter
alias. This changes the filter dispatcher only; weights, logits processors,
temperature, top-k/top-p, random sampling and RNG ownership stay intact.
The batched native helper avoids an additional Python loop per vocabulary row.
Actual C1/C8 probability equality and serving overhead need GPU qualification.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types


SOURCE_SHA256 = {
    'topk_topp_sampler.py': '60d816d829797436240bada454dd66d6ab583522656e5db5d6ed3342c109eefc',
    'topk_topp_triton.py': '255619f699898ddaaac959d9bef4494294c76d730abb8e90549afac83a9bcbe9',
    'rejection_sampler.py': '4bb87d7984d967be9c7e5b6a6b489c02571ef2f2913211cb3f1eea74b2e1f32f',
}
_SAMPLER = 'vllm.v1.sample.ops.topk_topp_sampler'
_REJECTION = 'vllm.v1.sample.rejection_sampler'
_MARKER = '_ciru_final_sampler_state'


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(evidence_dir=None):
    """Bind canonical filtering; call in Worker.load_model before warmup.

    The returned/printed receipt binds this adapter, all three runtime source
    files and the actual sampler/rejection aliases. An optional fresh evidence
    directory receives final-sampler-binding.json. Source drift or previously
    rebound native functions/required aliases fail before mutation.
    """
    sampler = importlib.import_module(_SAMPLER)
    rejection = importlib.import_module(_REJECTION)
    triton = importlib.import_module('vllm.v1.sample.ops.topk_topp_triton')
    modules = [sampler, triton, rejection]
    source_paths = {Path(module.__file__).name: Path(module.__file__).resolve()
                    for module in modules}
    actual = {name: _digest(path) for name, path in source_paths.items()}
    if actual != SOURCE_SHA256:
        raise RuntimeError('Final sampler requires the reviewed native source hashes: '+json.dumps(actual))

    native = sampler.apply_top_k_top_p_pytorch
    if (not isinstance(native, types.FunctionType)
            or native.__name__ != 'apply_top_k_top_p_pytorch'
            or native.__module__ != _SAMPLER
            or Path(native.__code__.co_filename).resolve() != source_paths['topk_topp_sampler.py']):
        raise RuntimeError('The pinned native PyTorch helper was rebound')
    state = getattr(sampler, _MARKER, None)
    if state is None:
        original = sampler.apply_top_k_top_p
        if (not isinstance(original, types.FunctionType)
                or original.__name__ != 'apply_top_k_top_p'
                or original.__module__ != _SAMPLER
                or Path(original.__code__.co_filename).resolve() != source_paths['topk_topp_sampler.py']):
            raise RuntimeError('Native filter dispatcher was rebound before installation')
        if rejection.apply_top_k_top_p is not original:
            raise RuntimeError('Rejection filter alias does not match the reviewed dispatcher')
    else:
        original = state['original']
        if (native is not state['native'] or sampler.apply_top_k_top_p is not native
                or rejection.apply_top_k_top_p not in (native, original)):
            raise RuntimeError('Installed canonical sampler or rejection alias was rebound')

    bindings = []
    for name, module in tuple(sys.modules.items()):
        if (not isinstance(module, types.ModuleType)
                or not name.startswith(('vllm.', 'ornith_g256.', 'tools.'))):
            continue
        for key, value in tuple(vars(module).items()):
            if value is original:
                bindings.append((module, key))
    if state is None and not all((module, 'apply_top_k_top_p') in bindings
                                 for module in (sampler, rejection)):
        raise RuntimeError('Required sampler/rejection bindings are missing')

    bound_names = sorted(set((state['receipt']['bound_aliases'] if state else [])
                            + [module.__name__+'.'+key for module, key in bindings]))
    receipt = {
        'schema': 'ciru.ornith.final_sampler.binding.v1',
        'pid': os.getpid(),
        'adapter_sha256': _digest(Path(__file__)),
        'canonical_backend': 'native-pytorch',
        'canonical_function': _SAMPLER+'.apply_top_k_top_p_pytorch',
        'runtime_source_sha256': actual,
        'runtime_source_paths': {name: str(path) for name, path in source_paths.items()},
        'bound_aliases': bound_names,
        'future_imports_use_canonical_module_binding': True,
        'native_pytorch_and_triton_helpers_unchanged': True,
        'batched_native_helper_without_row_loop': True,
        'install_before_compile_and_oracle_hooks': True,
        'rng_or_sampling_parameters_changed': False,
        'weights_or_target_logits_changed': False,
        'full_pq_or_serving_qualified': False,
        'promotion': False,
    }
    output_path = None if evidence_dir is None else Path(evidence_dir)/'final-sampler-binding.json'
    if output_path is not None and output_path.exists():
        if state is None or json.loads(output_path.read_text()) != receipt:
            raise RuntimeError('Refusing to replace a conflicting sampler receipt')

    for module, key in bindings:
        setattr(module, key, native)
    installed = {'original': original, 'native': native, 'receipt': receipt}
    setattr(sampler, _MARKER, installed)
    try:
        if output_path is not None and not output_path.exists():
            with output_path.open('x') as stream:
                stream.write(json.dumps(receipt, indent=2)+'\n')
    except Exception:
        for module, key in bindings:
            setattr(module, key, original)
        if state is None:
            delattr(sampler, _MARKER)
        else:
            setattr(sampler, _MARKER, state)
        raise
    print('ORNITH_FINAL_SAMPLER '+json.dumps(receipt, sort_keys=True), flush=True)
    return receipt
