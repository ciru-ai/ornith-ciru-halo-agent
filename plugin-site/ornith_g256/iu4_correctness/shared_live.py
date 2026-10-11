"""Repair shared decode arithmetic while preserving parent prefill outputs.

Small batches retain the parent's fused routed launch.  Its existing shared
buffer is then overwritten only at semantic decode rows.  Larger batches
retain the parent shared MLP output and similarly overwrite decode rows.
This deliberately duplicates shared work until correctness and serving cost
have been measured.  No checkpoint bank or production plugin is modified.
"""
import ctypes
import hashlib
import inspect
import json
from pathlib import Path

import torch

from ornith_g256.iu4_correctness.shared_kernel import fixed8


def install(worker, evidence_dir):
    from ornith_g256 import dense_n32, moe_shared
    from ornith_g256.method import G256MoEMethod, G256LinearMethod
    from ornith_g256.iu4_correctness.gates_phase import segments_for

    root = Path(evidence_dir)
    root.mkdir(parents=True, exist_ok=True)
    if getattr(worker, '_iu4_shared_installed', False):
        raise RuntimeError('IU4 shared correctness adapter installed twice')
    if not moe_shared._INSTALLED or not dense_n32._INSTALLED:
        raise RuntimeError('Shared adapter must follow installed IU4 model loading')
    source_path = Path(inspect.getsourcefile(moe_shared))
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    expected = 'b1ba457cc20d6ad150dc1fc6854e73365a7476c80d57e687dec7d085d47b98c1'
    if digest != expected:
        raise RuntimeError('Installed fused shared source differs from reviewed v4.1.2')
    rows_query = dense_n32._LIB.ornith_dense_g256_n32_rows
    rows_query.argtypes, rows_query.restype = [], ctypes.c_int
    if rows_query() != 16:
        raise RuntimeError('Fixed8 adapter requires qualified N32 rows1–16 ABI')
    native_path = Path(dense_n32._LIB._name).resolve(strict=True)
    native_digest = hashlib.sha256(native_path.read_bytes()).hexdigest()
    expected_native = '504b832eea9279f98be4eab3342f0ddd31a5b3d191235f856acaf1c1b0e970c9'
    if native_digest != expected_native:
        raise RuntimeError('Installed dense N32 binary differs from reviewed v4.1.2')

    selected = {}
    names = []
    for name, layer in worker.model_runner.get_model().named_modules():
        if '.layers.' not in name or not name.endswith('.mlp.shared_expert'):
            continue
        gu, down = layer.gate_up_proj, layer.down_proj
        assert isinstance(gu.quant_method, G256LinearMethod)
        assert isinstance(down.quant_method, G256LinearMethod)
        assert (gu.quant_method.n, gu.quant_method.k) == (1024, 2048)
        assert (down.quant_method.n, down.quant_method.k) == (2048, 512)
        assert tuple(layer.expert_gate.weight.shape) == (1, 2048)
        for linear in (gu, down):
            assert hasattr(linear, '_dense_n32_codes') and hasattr(linear, '_source_prefill_weight')
            assert linear._g256_ready and linear.quant_method.runtime is not None
        selected[id(layer)] = layer
        names.append(name)
    assert len(selected) == 40, len(selected)
    seen = set()
    counters = {'fused_replacement_calls': 0, 'unfused_replacement_calls': 0}

    @torch.library.custom_op('ornith_iu4_shared_fixed8::repair',
                            mutates_args={'output', 'workspace', 'gate_flags', 'down_flags'},
                            device_types='cuda')
    def repair(x: torch.Tensor, output: torch.Tensor,
               gate_codes: torch.Tensor, gate_meta: torch.Tensor,
               down_codes: torch.Tensor, down_meta: torch.Tensor,
               scalar_weight: torch.Tensor, workspace: torch.Tensor,
               gate_flags: torch.Tensor, down_flags: torch.Tensor,
               capacity: int, fused_parent: bool) -> None:
        segments = segments_for(x.shape[0])
        decode_rows = sum(end - first for first, end, prefill in segments if not prefill)
        native_launch_pairs = sum((end - first + 7) // 8 for first, end, prefill in segments if not prefill)
        for first, end, prefill in segments:
            if not prefill and end > first:
                output[first:end].copy_(fixed8(
                    x[first:end], gate_codes, gate_meta, down_codes, down_meta,
                    scalar_weight, workspace, gate_flags, down_flags, capacity))
        capture = torch.cuda.is_current_stream_capturing()
        signature = (x.shape[0], tuple(segments), fused_parent, capture)
        if signature not in seen:
            seen.add(signature)
            with (root / 'shared-dispatch.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(
                    rows=x.shape[0], segments=segments, fused_parent=fused_parent,
                    capture=capture, fixed_decode_rows=8,
                    semantic_decode_rows=decode_rows,
                    native_launch_pairs=native_launch_pairs,
                    native_dispatch_executed=decode_rows > 0,
                    native_symbol='ornith_dense_g256_launch_n32',
                    gate_up_shape=[1024, 2048], down_shape=[2048, 512],
                    activation_bits=8, weight_format='unchanged resident G256 IU4 N32 banks',
                    scalar_gate='rowwise FP32 sum, original BF16 weight/output',
                    prefill_action='existing parent output retained byte for byte',
                    duplicate_shared_work=True,
                    gpu_dispatch_collector_join_qualified=False)) + '\n')
        key = 'fused_replacement_calls' if fused_parent else 'unfused_replacement_calls'
        counters[key] += int(decode_rows > 0)

    @repair.register_fake
    def repair_fake(x, output, gate_codes, gate_meta, down_codes, down_meta,
                    scalar_weight, workspace, gate_flags, down_flags, capacity,
                    fused_parent):
        return None

    def replace(layer, x, output, fused):
        gu, down = layer.gate_up_proj, layer.down_proj
        gr, dr = gu.quant_method, down.quant_method
        assert gr.runtime is dr.runtime
        repair(x, output, gu._dense_n32_codes, gu._dense_n32_metadata,
               down._dense_n32_codes, down._dense_n32_metadata,
               layer.expert_gate.weight, gr.runtime.workspace,
               gr.runtime.slots[gr.slot], dr.runtime.slots[dr.slot],
               gr.runtime.capacity, fused)

    original_moe_apply = G256MoEMethod.apply
    closure = inspect.getclosurevars(original_moe_apply).nonlocals
    if 'original_apply' not in closure or '_STATE' not in original_moe_apply.__globals__:
        raise RuntimeError('The expected v4.1.2 fused-shared wrapper is inactive')

    def moe_apply(method, layer, x, topk_weights, topk_ids,
                  shared_experts=None, shared_experts_input=None):
        shared_layer = None if shared_experts is None else shared_experts._layer
        pending = moe_shared._STATE.get('pending')
        result = original_moe_apply(method, layer, x, topk_weights, topk_ids,
                                    shared_experts, shared_experts_input)
        # The pending buffer is consumed only when the parent's fused launch
        # actually ran.  We preserve its routed result and all prefill rows.
        if (id(shared_layer) in selected and pending is not None
                and moe_shared._STATE.get('pending') is not pending):
            assert shared_experts_input is not None
            replace(shared_layer, shared_experts_input, pending, True)
        return result

    G256MoEMethod.apply = moe_apply
    mlp_class = type(next(iter(selected.values())))
    original_mlp_forward = mlp_class.forward

    def mlp_forward(layer, x):
        output = original_mlp_forward(layer, x)
        if id(layer) in selected:
            replace(layer, x, output, False)
        return output

    mlp_class.forward = mlp_forward
    worker._iu4_shared_installed = True
    worker._iu4_shared_counters = counters
    worker._iu4_shared_layers = selected
    worker._iu4_shared_replace = replace
    worker._iu4_shared_original_mlp_forward = original_mlp_forward
    (root / 'shared-binding.json').write_text(json.dumps(dict(
        candidate='IU4 shared fixed8 quant-preserving decode repair', selected=names,
        shared_experts=40, packed_matrices=80,
        checkpoint_banks_changed=False, bf16_weight_replacement=False,
        additional_weight_residency_bytes=0, decode_chunk_rows=8,
        source_parent_sha256=digest,
        native_library_path=str(native_path), native_library_sha256=native_digest,
        native_symbol='ornith_dense_g256_launch_n32', native_reported_rows=16,
        arithmetic_delta='Semantic decode shared gate/up and down always use the existing N32 M8 A8/IU4 specialization; scalar gate is rowwise FP32 with original BF16 output. Shared SiLU/mul and sigmoid use the ordinary vLLM/PyTorch BF16 contract.',
        preserved_prefill='Both fused small-row and unfused large-row parent shared results are computed first; only semantic decode rows are overwritten.',
        cost='Parent shared calculation remains and candidate shared decode calculation is additional work.',
        actual_batch_invariance_qualified=False, teacher_quality_qualified=False,
        served_performance_qualified=False, promotion=False), indent=2) + '\n')
    return dict(selected=40, matrices=80, packed_weights_unchanged=True,
                duplicate_shared_work=True)
