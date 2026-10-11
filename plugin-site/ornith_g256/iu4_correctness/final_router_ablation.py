"""Source-bound component ablations; original router prompt score values.

No weights, native files, sampler or policy are modified here. Router-decode
keeps the parent's full-M prompt reduction and copies its scores exactly to
FP32; semantic decode rows keep the frozen rowwise FP32 reduction. Downstream
model/prefill equality still requires the separate live numerical capture.
"""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re


FROZEN_SOURCES = {
    'gates.py': '3c729a9006fc925aef16cb97aab9e198ab12d2c3dacb9a84aee6163f466641d4',
    'gates_phase.py': '9e224fbd907c97120c84baedd68988783679edb32227c8c0fe750fc94e7bc2cf',
    'gates_projection.py': '5558c0edcbd0a424aae55b7e383c482ffa47ba21a6f584ee041c7d73999f8c98',
    'shared_live.py': 'edb525c42b81c902fda88881cdb74b9464acbd67e8b2c3b9a440692170626dfd',
    'shared_kernel.py': 'c6395f0d7835976e59384e9e867ff6274ef4fb8dbef5067ece53153fe029fd76',
    'rng_exact_restore.py': 'a25537137fb6dac8fc461a2712971c79ad33753a727241d4f8c7f77dfd03adf1',
}


def variant_plan(variant):
    plans = {
        'all': (True, 'all', True),
        'router-decode': (True, 'decode', True),
        'ba-only': (True, 'off', False),
        'router-only': (False, 'all', False),
        'shared-only': (False, 'off', True),
        'no-shared': (True, 'all', False),
        'no-router': (True, 'off', True),
    }
    if variant not in plans:
        raise ValueError('Unknown final numerical variant: ' + str(variant))
    ba, router, shared = plans[variant]
    return dict(ba=ba, router=router, shared=shared)


def read_config(environ):
    arm = environ.get('IU4_ARM')
    if arm not in ('baseline', 'candidate'):
        raise ValueError('IU4_ARM must be baseline or candidate')
    variant = environ.get('IU4_FINAL_VARIANT', 'all')
    repairs = variant_plan(variant)
    sampler = environ.get('IU4_FINAL_SAMPLER', '1')
    if sampler not in ('0', '1'):
        raise ValueError('IU4_FINAL_SAMPLER must be 0 or 1')
    mixed = environ.get('IU4_FINAL_MIXED', '0')
    if mixed not in ('0', '1'):
        raise ValueError('IU4_FINAL_MIXED must be 0 or 1')
    native_sampling = environ.get('IU4_FINAL_NATIVE_SAMPLING', '1')
    if native_sampling not in ('0', '1'):
        raise ValueError('IU4_FINAL_NATIVE_SAMPLING must be 0 or 1')
    if arm == 'candidate' and native_sampling == '1' and sampler != '1':
        raise ValueError('Native sampling requires the canonical filter')
    if arm == 'baseline':
        repairs = dict(ba=False, router='off', shared=False)
    return dict(arm=arm, variant=variant, repairs=repairs,
                sampler=arm == 'candidate' and sampler == '1',
                native_sampling=arm == 'candidate' and native_sampling == '1',
                mixed=arm == 'candidate' and mixed == '1')


def verify_frozen_sources():
    actual = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in FROZEN_SOURCES}
    if actual != FROZEN_SOURCES:
        raise RuntimeError('Final numerical ablation frozen components differ')
    return actual


def route_scores(x, segments, parent, decode, allocate):
    """Preserve full-M prompt geometry, replacing only admitted decode slices.

    Callbacks return FP32 scores; parent casts its original BF16 values exactly.
    Dependencies are injected so CPU tests exercise dispatch without Torch/GPU.
    """
    rows, previous = x.shape[0], 0
    if rows <= 0 or not segments:
        raise ValueError('Empty router phase plan')
    for first, end, prefill in segments:
        if first != previous or not first < end <= rows or type(prefill) is not bool:
            raise ValueError('Invalid router phase plan')
        previous = end
    if previous != rows:
        raise ValueError('Router phase plan omits rows')
    out = parent(x) if any(prefill for _, _, prefill in segments) else allocate(rows)
    for first, end, prefill in segments:
        if not prefill:
            out[first:end].copy_(decode(x[first:end]))
    return out


def install_components(worker, evidence_dir, config):
    if getattr(worker, '_final_components_installed', False):
        raise RuntimeError('Final numerical components installed twice')
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from ornith_g256.iu4_correctness.gates_phase import install as install_phase, segments_for
    from ornith_g256.iu4_correctness.gates_projection import router

    hashes = verify_frozen_sources()
    plan = config['repairs']
    root = Path(evidence_dir)
    root.mkdir(parents=True, exist_ok=True)
    install_phase()
    if plan['ba'] or plan['router'] != 'off':
        from ornith_g256.iu4_correctness.gates import install
        install(worker, root)
        ordinary = worker._iu4_gates_original_apply
        repaired = UnquantizedLinearMethod.apply
        ba_ids, router_ids, router_layers = set(), {}, {}
        for name, layer in worker.model_runner.get_model().named_modules():
            if '.layers.' not in name:
                continue
            if name.endswith('.linear_attn.in_proj_ba'):
                ba_ids.add(id(layer))
            elif name.endswith('.mlp.gate'):
                match = re.search(r'\.layers\.(\d+)\.mlp\.gate$', name)
                if match is None:
                    raise RuntimeError('Router has no stable architecture layer number')
                key = int(match.group(1))
                if key in router_layers:
                    raise RuntimeError('Duplicate router architecture layer number')
                router_ids[id(layer)] = key
                router_layers[key] = (name, layer)
        assert len(ba_ids) == 30 and set(router_layers) == set(range(40))
        seen = set()

        if plan['router'] == 'decode':
            @torch.library.custom_op('ornith_iu4_final::router_decode', mutates_args=(), device_types='cuda')
            def router_decode(x: torch.Tensor, weight: torch.Tensor, layer_key: int) -> torch.Tensor:
                name, layer = router_layers[layer_key]
                assert weight.data_ptr() == layer.weight.data_ptr()
                segments = segments_for(x.shape[0])
                out = route_scores(x, segments,
                    lambda value: ordinary(layer.quant_method, layer, value, None).to(torch.float32),
                    lambda value: router(value, weight)[0],
                    lambda rows: torch.empty((rows, 256), device=x.device, dtype=torch.float32))
                signature = (x.shape[0], tuple(segments), torch.cuda.is_current_stream_capturing())
                if signature not in seen:
                    seen.add(signature)
                    receipt = dict(pid=os.getpid(), name=name, stable_layer_key=layer_key,
                        rows=x.shape[0], segments=segments, capture=signature[-1],
                        prefill='Original full-M scores, exact BF16-to-FP32 value cast',
                        decode='Frozen rowwise FP32 products/reduction/scores',
                        original_bf16_weights=True, actual_custom_op_invocation=True)
                    with (root / 'final-router-launches.jsonl').open('a') as stream:
                        stream.write(json.dumps(receipt) + '\n')
                    print('ORNITH_FINAL_ROUTER_DISPATCH ' + json.dumps(receipt), flush=True)
                return out

            @router_decode.register_fake
            def router_fake(x, weight, layer_key):
                return torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)

        @wraps(repaired)
        def apply(method, layer, x, bias=None):
            key = id(layer)
            if key in ba_ids and not plan['ba']:
                return ordinary(method, layer, x, bias)
            if key in router_ids:
                if plan['router'] == 'off':
                    return ordinary(method, layer, x, bias)
                if plan['router'] == 'decode':
                    assert bias is None
                    return router_decode(x, layer.weight, router_ids[key])
            return repaired(method, layer, x, bias)

        UnquantizedLinearMethod.apply = apply
    if plan['shared']:
        from ornith_g256.iu4_correctness.shared_live import install
        install(worker, root)
    binding = dict(pid=os.getpid(), variant=config['variant'], repairs=plan,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_sources=hashes, original_weights_unchanged=True, added_weight_bytes=0,
        router_prefill_full_model_equivalence_qualified=False,
        numerical_quality_qualified=False, concurrent_fidelity_qualified=False, promotion=False)
    (root / 'final-components-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    print('ORNITH_FINAL_COMPONENTS_INSTALLED ' + json.dumps(binding), flush=True)
    worker._final_components_installed = True
    return binding
