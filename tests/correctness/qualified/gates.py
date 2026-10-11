"""Load-time GDN BA/router repair for an isolated Ornith IU4 worker.

All original weights stay in place. BA prompt rows retain the actual released
full-M UnquantizedLinearMethod backend, including mixed-batch geometry. The
layer-1 exception retains IU4's original M1 backend separately for each decode
row; the other 29 GDN layers use the screened rowwise FP32 sum. Router scores
are FP32 rowwise for all shapes, as in the Ultra correction.
"""
import hashlib
import json
import re
from pathlib import Path

import torch

from tools.gates_phase import install as install_phase, segments_for
from tools.gates_projection import ba_into, router


def _kernel_receipt(kernel):
    code = kernel.asm.get("hsaco")
    if isinstance(code, str):
        code = Path(code).read_bytes()
    return dict(kernel=kernel.name,
        code_sha256=hashlib.sha256(code).hexdigest() if code else None,
        registers=kernel.n_regs, spills=kernel.n_spills)


def install(worker, evidence_dir):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm import envs
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    install_phase()
    model = worker.model_runner.get_model()
    ba_layers, ba_keys, router_layers = {}, {}, {}
    for name, layer in model.named_modules():
        if ".layers." not in name:
            continue
        target = ba_layers if name.endswith(".linear_attn.in_proj_ba") else (
            router_layers if name.endswith(".mlp.gate") else None)
        if target is None:
            continue
        assert isinstance(layer.quant_method, UnquantizedLinearMethod), name
        assert layer.weight.dtype == torch.bfloat16 and layer.weight.is_contiguous(), name
        assert getattr(layer, "bias", None) is None, name
        expected = (64, 2048) if target is ba_layers else (256, 2048)
        assert tuple(layer.weight.shape) == expected, (name, layer.weight.shape)
        if target is ba_layers:
            match = re.search(r"\.layers\.(\d+)\.linear_attn\.in_proj_ba$", name)
            assert match is not None, name
            stable_key = int(match.group(1))
            assert stable_key not in ba_layers, (name, stable_key)
            ba_layers[stable_key] = (name, layer)
            ba_keys[id(layer)] = stable_key
        else:
            target[id(layer)] = (name, layer)
    assert len(ba_layers) == 30 and len(router_layers) == 40
    assert set(ba_layers) == {number for number in range(40) if number % 4 != 3}
    layer1 = 1
    ordinary = UnquantizedLinearMethod.apply
    seen = set()

    def record(kind, signature, payload):
        key = (kind, signature)
        if key in seen:
            return
        seen.add(key)
        payload.update(kind=kind, evidence="Actual custom-op invocation/capture; GPU optimizer qualification is separate")
        with (evidence / "gates-launches.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
        print("IU4_CORRECTNESS_GATES_DISPATCH " + json.dumps(payload), flush=True)

    @torch.library.custom_op("ornith_iu4_correctness::ba", mutates_args=(), device_types="cuda")
    def ba_op(x: torch.Tensor, weight: torch.Tensor, layer_key: int) -> torch.Tensor:
        name, layer = ba_layers[layer_key]
        assert weight.data_ptr() == layer.weight.data_ptr()
        segments = segments_for(x.shape[0])
        # Parent full-M output preserves prompt reduction geometry in mixed
        # batches. Only semantic decode slices are replaced afterward.
        has_prefill = any(prefill for _, _, prefill in segments)
        out = ordinary(layer.quant_method, layer, x, None) if has_prefill else x.new_empty((x.shape[0], 64))
        capturing = torch.cuda.is_current_stream_capturing()
        for first, end, prefill in segments:
            if prefill:
                continue
            if layer_key == layer1:
                # Use the actual installed original backend, which can differ
                # from torch.mm on ROCm. Each call has original M1 geometry.
                for row in range(first, end):
                    out[row:row + 1].copy_(ordinary(layer.quant_method, layer, x[row:row + 1], None))
                receipt = dict(operator="Original IU4 M1 UnquantizedLinearMethod.apply per decode row",
                    backend=getattr(layer.quant_method._gemm_impl, "__qualname__", str(layer.quant_method._gemm_impl)))
            else:
                kernel = ba_into(x[first:end], weight, out[first:end])
                receipt = dict(operator="Rowwise FP32 products/sum, BF16 output", **_kernel_receipt(kernel))
            record("ba-layer1" if layer_key == layer1 else "ba-rowwise",
                (x.shape[0], first, end, capturing), dict(name=name, original_rows=x.shape[0], rows=end-first,
                    first=first, segments=segments, capture=capturing,
                    input_stride=list(x[first:end].stride()), weight_shape=list(weight.shape),
                    original_bf16_weights=True, retained_full_m_prefill=has_prefill, **receipt))
        if has_prefill:
            record("ba-prefill", (x.shape[0], tuple(segments), capturing), dict(name=name, rows=x.shape[0],
                segments=segments, capture=capturing, operator="Original full-M UnquantizedLinearMethod.apply",
                backend=getattr(layer.quant_method._gemm_impl, "__qualname__", str(layer.quant_method._gemm_impl))))
        return out

    @ba_op.register_fake
    def ba_fake(x, weight, layer_key):
        return x.new_empty((x.shape[0], 64))

    @torch.library.custom_op("ornith_iu4_correctness::router", mutates_args=(), device_types="cuda")
    def router_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out, kernel = router(x, weight)
        capturing = torch.cuda.is_current_stream_capturing()
        record("router", (x.shape[0], x.stride(0), capturing), dict(rows=x.shape[0],
            input_stride=list(x.stride()), operand_dtype=str(x.dtype), score_dtype=str(out.dtype),
            capture=capturing, operator="Uniform rowwise FP32 router reduction, all target shapes",
            original_bf16_weights=True, **_kernel_receipt(kernel)))
        return out

    @router_op.register_fake
    def router_fake(x, weight):
        return torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)

    def apply(method, layer, x, bias=None):
        key = id(layer)
        if key in ba_keys:
            assert bias is None
            # This scalar is serialized in persistent AOT graphs. A process
            # memory address is invalid after worker reload; layer number is
            # stable across workers using this same model architecture.
            return ba_op(x, layer.weight, ba_keys[key])
        if key in router_layers:
            assert bias is None
            return router_op(x, layer.weight)
        return ordinary(method, layer, x, bias)

    UnquantizedLinearMethod.apply = apply
    worker._iu4_gates_original_apply = ordinary
    worker._iu4_gates_installed = True
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("gates.py", "gates_phase.py", "gates_projection.py")}
    binding = dict(candidate="ornith-iu4-correctness-gates-v2", sources=sources,
        ba_selected=[name for name, _ in ba_layers.values()],
        router_selected=[name for name, _ in router_layers.values()],
        ba_decode="29 rowwise FP32 layers; layer 1 retains actual IU4 original M1 backend for each row",
        ba_prefill="Original UnquantizedLinearMethod.apply full-M, including mixed geometry",
        router="All-shape rowwise FP32 products/reduction/scores; original BF16 weights",
        batch_invariant_env=envs.VLLM_BATCH_INVARIANT,
        original_weights_unchanged=True, added_weight_bytes=0,
        semantic_phase_source="Synchronous prepared CPU request plan and uniform FULL capture descriptor",
        aot_ba_identity="Stable architecture layer number 0..39; process-local module IDs never enter custom-op graph arguments",
        natural_quality_qualified=False, serving_performance_qualified=False,
        full_optimizer_qualified=False, promotion=False)
    (evidence / "gates-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    print("IU4_CORRECTNESS_GATES_BINDING " + json.dumps(binding), flush=True)
    return binding
