"""Isolated K3/Q4 admission for the exact released IU4 graph/boundary hooks.

Install in each process after native package import and before worker load or
graph construction. Capacity remains DF15/7; only the current C1 width gains Q4.
This adapter does not change sampling, numerical kernels, or global graph sizes.
"""
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import textwrap


GRAPH_SHA256 = 'fa4ddbb0609ed4ba0065b9eb2753a7a4a9f79edb37cb75ea875973b9fb0eab67'
BOUNDARY_SHA256 = 'a5f46458194cade5ddcbfbb51af30c662d12b9086fc459ecbeb17463339febb1'
_Q4_ALLOWED = ContextVar('ornith_iu4_k3_q4_decode_allowed', default=True)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_contract(graphs, boundary):
    for module, expected in ((graphs, GRAPH_SHA256), (boundary, BOUNDARY_SHA256)):
        actual = _sha256(inspect.getfile(module))
        if actual != expected:
            raise RuntimeError(f'IU4 K3 geometry source mismatch: {module.__name__}: {actual}')
    return {
        'pid': os.getpid(),
        'adapter_sha256': _sha256(__file__),
        'dynamic_graphs_sha256': GRAPH_SHA256,
        'dflash_conv_boundary_sha256': BOUNDARY_SHA256,
        'added_draft_depth': 3,
        'added_target_query_width': 4,
        'concurrency': 1,
        'allocated_capacity_depth': 15,
        'numerical_changes': False,
        'actual_dispatch_verified': False,
    }


def _install_dispatch_proof(graphs, contract):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._determine_batch_execution_and_padding
    if getattr(original, '_iu4_k3_geometry_dispatch', False):
        return

    @wraps(original)
    def dispatch(runner, num_tokens, num_reqs, counts, maximum, *args, **kwargs):
        q4 = num_tokens == 4 and num_reqs == 1 and maximum == 4
        live = kwargs.get('force_uniform_decode') is None
        decode = False
        if q4 and live:
            batch = runner.input_batch
            decode = (batch.num_reqs == 1 and
                      batch.num_computed_tokens_cpu[0] >= batch.num_prompt_tokens[0])
        token = _Q4_ALLOWED.set(not (q4 and live) or decode)
        try:
            result = original(runner, num_tokens, num_reqs, counts, maximum, *args, **kwargs)
        finally:
            _Q4_ALLOWED.reset(token)
        if (q4 and live and decode and result[0].name == 'FULL'
                and graphs._enabled(runner)
                and not getattr(runner, '_iu4_k3_q4_dispatch_logged', False)):
            print('ORNITH_IU4_K3_Q4_DISPATCH ' + json.dumps({
                **contract, 'pid': os.getpid(), 'num_tokens': 4, 'num_reqs': 1,
                'max_query_len': 4, 'semantic_decode': True, 'runtime_mode': 'FULL',
                'actual_dispatch_verified': True,
            }, sort_keys=True), flush=True)
            runner._iu4_k3_q4_dispatch_logged = True
        return result

    dispatch._iu4_k3_geometry_dispatch = True
    GPUModelRunner._determine_batch_execution_and_padding = dispatch


def install_modules(graphs, boundary):
    """Admit Q4 on verified native modules before the boundary load hook installs."""
    contract = _source_contract(graphs, boundary)
    if getattr(graphs, '_iu4_k3_geometry_installed', False):
        return contract
    if boundary._INSTALLED:
        raise RuntimeError('IU4 K3 geometry must install before the boundary load hooks')

    # Recompile only the guarded Python install closure, keeping its exact globals.
    # The capacity/load guard stays Q8/Q16. Q4 is a power of two, so the released
    # dynamic mask kernel and convolution retain the same bitmask boundary rule.
    original_guard = ("if width not in (8, 16):\n"
                      "            raise ValueError(f'Unexpected current DFlash query width{width}')")
    source = textwrap.dedent(inspect.getsource(boundary.install))
    if source.count(original_guard) != 1:
        raise RuntimeError('IU4 K3 geometry could not identify the current-width boundary guard')
    replacement = source.replace(original_guard, original_guard.replace('(8, 16)', '(4, 8, 16)'))
    staged = {}
    exec(compile(replacement, inspect.getfile(boundary), 'exec'), boundary.__dict__, staged)
    new_boundary_install = staged['install']

    original_supported = graphs._supported_query_len
    original_replace = graphs._replace_full_keys
    original_install = graphs.install

    @wraps(original_supported)
    def supported(tokens, requests, maximum):
        if tokens == 4 and requests == 1 and maximum == 4:
            return 4 if _Q4_ALLOWED.get() else None
        return original_supported(tokens, requests, maximum)

    @wraps(original_replace)
    def replace_keys(runner):
        original_replace(runner)
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor

        dispatcher = runner.cudagraph_dispatcher
        dispatcher._bs_to_padded_graph_size[4] = 4
        dispatcher.cudagraph_keys[CUDAGraphMode.FULL].add(
            BatchDescriptor(num_tokens=4, num_reqs=1, uniform=True))

    @wraps(original_install)
    def install_graphs():
        result = original_install()
        _install_dispatch_proof(graphs, contract)
        return result

    boundary.install = new_boundary_install
    graphs._supported_query_len = supported
    graphs._replace_full_keys = replace_keys
    graphs.install = install_graphs
    graphs._iu4_k3_geometry_installed = True
    if graphs._INSTALLED:
        _install_dispatch_proof(graphs, contract)
    print('ORNITH_IU4_K3_GEOMETRY ' + json.dumps(contract, sort_keys=True), flush=True)
    return contract


def install():
    """Called independently by parent and multiprocessing engine entrypoints."""
    return install_modules(importlib.import_module('ornith_g256.dynamic_graphs'),
                           importlib.import_module('ornith_g256.dflash_conv_boundary'))
