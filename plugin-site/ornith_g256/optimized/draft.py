"""Isolated draft residual/RMSNorm repair; public target and sampler preserved."""
import ast
import hashlib
import json
import os
from pathlib import Path
import torch
from ornith_g256.worker import OrnithG256Worker as Parent
from .draft_kernel import launch

R = Path(__file__).resolve().parents[1]
NAME = 'triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_3'
KINDS = {
    '3945f633918257a81244518749ca8c2613735dbedb656d49d811eba9e07a5f9a': 'middle',
    '456e4167b1199b272395eae07c665ad6b501b39cb0018097a4abf87c72861bd7': 'final',
}

def install(output, cache_root):
    out = Path(output)
    state = dict(candidate_installed=True, kernels=[], calls={}, captured_rows={},
                 graph_replay_poison_check=False, sampler_changed=False)
    def persist():
        temp = out / 'kernel-provenance.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / 'kernel-provenance.json')
    # Compile all configured graph shapes before graph capture, without RNG use.
    for rows in [1, 8, 16, 32, 48, 64]:
        residual = torch.zeros((rows, 2048), device='cuda', dtype=torch.float32)
        base = torch.zeros((2, 2, 2048), device='cuda', dtype=torch.bfloat16)
        coef = torch.zeros((rows, 512), device='cuda', dtype=torch.bfloat16)
        hidden = torch.zeros_like(residual, dtype=torch.bfloat16)
        mask = torch.full((1,), 15, device='cuda', dtype=torch.int32)
        weight = torch.ones((2048,), device='cuda', dtype=torch.bfloat16)
        out_residual = torch.empty_like(hidden)
        out_norm = torch.empty_like(hidden)
        for kind in ['middle', 'final']:
            inputs = [residual, base, coef, hidden, mask, weight]
            if kind == 'middle':
                inputs.append(out_residual)
            inputs.extend([out_norm, rows, 2048])
            launch(kind, inputs, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(kind, inputs, torch.cuda.current_stream().cuda_stream)
            out_norm.fill_(float('nan'))
            out_residual.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(out_norm) == 0
            if kind == 'middle':
                assert torch.count_nonzero(out_residual) == 0
            del graph
    state['graph_replay_poison_check'] = True
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    original = CachingAutotuner.run
    classified, seen = {}, set()
    draft_root = str(Path(cache_root) / 'vllm/torch_compile_cache/torch_aot_compile')

    def run(kernel, *inputs, stream, benchmark_run=False, **kwargs):
        if kernel not in classified:
            kind = None
            if kernel.fn.__name__ == NAME:
                relative = Path(kernel.filename).relative_to(draft_root)
                assert len(relative.parts[0]) == 64 and 'inductor_cache' in relative.parts, kernel.filename
                fn = next(n for n in ast.parse(kernel.fn.src).body if isinstance(n, ast.FunctionDef))
                fn.decorator_list = []
                digest = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
                assert digest in KINDS, digest
                kind = KINDS[digest]
                state['kernels'].append(dict(kind=kind, function=NAME, filename=kernel.filename, ast_sha256=digest))
            classified[kernel] = kind
        kind = classified[kernel]
        if kind is None:
            return original(kernel, *inputs, stream=stream, benchmark_run=benchmark_run, **kwargs)
        assert not benchmark_run and not kwargs
        launch(kind, inputs, stream)
        rows = int(inputs[-2])
        captured = torch.cuda.is_current_stream_capturing()
        key = kind + ':' + str(rows)
        state['calls'][key] = state['calls'].get(key, 0) + 1
        if captured and rows not in state['captured_rows'].setdefault(kind, []):
            state['captured_rows'][kind].append(rows)
        if (kind, rows, captured) not in seen:
            seen.add((kind, rows, captured))
            persist()
        return None

    CachingAutotuner.run = run
    persist()
