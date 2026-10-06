"""Apply the verified store/reload barrier to the identified first draft norm."""
import ast
import hashlib
import json
from pathlib import Path

import torch

NAME = 'triton_red_fused_add_arange_bitwise_and_constant_pad_nd_fused_add_rms_norm_ge_mul_select_slice_unsqueeze_view_0'
ORIGINAL_AST = '119c21631928d7a4f9b57bcce5d7e28e10091b8ccf33a5159ac13b468c667401'
BARRIER_SHA = '9321a3721076171377f610c03870ea794e7239fc8159a6fad754061b4be21644'


def install(output, cache_root):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from . import norm0_barrier

    source = Path(norm0_barrier.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == BARRIER_SHA
    barrier = getattr(norm0_barrier, NAME)
    original_run = CachingAutotuner.run
    root = Path(cache_root) / 'vllm/torch_compile_cache/torch_aot_compile'
    state = dict(barrier_source_sha256=BARRIER_SHA, original_ast_sha256=ORIGINAL_AST,
                 kernels=[], calls={}, captured_rows=[], preflight_rows=[])
    output = Path(output)

    def persist():
        temporary = output / 'first-norm-provenance.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(output / 'first-norm-provenance.json')

    # Compile before capture and check store/reload behavior on every C8 shape.
    stream = torch.cuda.current_stream().cuda_stream
    for rows in [1, 8, 16, 24, 32, 40, 48, 56, 64]:
        base = torch.zeros((2, 2, 2048), device='cuda', dtype=torch.bfloat16)
        coefficients = torch.zeros((rows, 512), device='cuda', dtype=torch.bfloat16)
        hidden = torch.zeros((rows, 2048), device='cuda', dtype=torch.bfloat16)
        mask = torch.full((1,), 15, device='cuda', dtype=torch.int32)
        residual = torch.zeros_like(hidden)
        weight = torch.ones((2048,), device='cuda', dtype=torch.bfloat16)
        scratch = torch.empty_like(hidden, dtype=torch.float32)
        normalized = torch.empty_like(hidden)
        inputs = (base, coefficients, hidden, mask, residual, weight,
                  scratch, normalized, rows, 2048)
        original_run(barrier, *inputs, stream=stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            original_run(barrier, *inputs, stream=torch.cuda.current_stream().cuda_stream)
        scratch.fill_(float('nan'))
        normalized.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(scratch).item() == 0
        assert torch.count_nonzero(normalized).item() == 0
        state['preflight_rows'].append(rows)
        del graph

    classified, seen = {}, set()

    def run(kernel, *inputs, stream, benchmark_run=False, **kwargs):
        if kernel not in classified:
            selected = kernel.fn.__name__ == NAME
            if selected:
                relative = Path(kernel.filename).relative_to(root)
                assert len(relative.parts[0]) == 64 and 'inductor_cache' in relative.parts
                fn = next(n for n in ast.parse(kernel.fn.src).body if isinstance(n, ast.FunctionDef))
                fn.decorator_list = []
                digest = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
                assert digest == ORIGINAL_AST, ('Unrecognized first draft normalization', digest)
                state['kernels'].append(dict(filename=kernel.filename, ast_sha256=digest))
            classified[kernel] = selected
        if not classified[kernel]:
            return original_run(kernel, *inputs, stream=stream,
                                benchmark_run=benchmark_run, **kwargs)
        assert not benchmark_run and not kwargs
        assert len(inputs) == 10 and int(inputs[-1]) == 2048
        result = original_run(barrier, *inputs, stream=stream)
        rows = int(inputs[-2])
        capturing = torch.cuda.is_current_stream_capturing()
        state['calls'][str(rows)] = state['calls'].get(str(rows), 0) + 1
        if capturing and rows not in state['captured_rows']:
            state['captured_rows'].append(rows)
        if (rows, capturing) not in seen:
            seen.add((rows, capturing))
            persist()
        return result

    CachingAutotuner.run = run
    persist()
