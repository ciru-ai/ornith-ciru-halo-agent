"""Quantization-preserving, fixed-eight-row shared expert arithmetic.

The resident IU4 N32 banks, row-local Hadamard/A8 transform and BF16
intermediate/output contract are retained.  Every decode chunk executes the
same N32 M8 specialization, including K0/M1 and partial verification chunks.
This module does not select semantic phases or patch the installed package.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def scalar_gate(x, weight, out, M: tl.constexpr, K: tl.constexpr,
                STRIDE: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    values = tl.load(x + row * STRIDE + cols, mask=cols < K, other=0).to(tl.float32)
    w = tl.load(weight + cols, mask=cols < K, other=0).to(tl.float32)
    total = tl.sum(values * w, axis=0)
    tl.store(out + row, total, mask=row < M)


def gate(x, weight):
    assert x.ndim == 2 and x.shape[1] == 2048 and x.stride(1) == 1
    assert x.dtype == weight.dtype == torch.bfloat16
    assert tuple(weight.shape) == (1, 2048) and weight.is_contiguous()
    out = x.new_empty((x.shape[0], 1))
    scalar_gate[(x.shape[0],)](x, weight, out, x.shape[0], 2048, x.stride(0),
                             2048, num_warps=4)
    return out


def fixed8(x, gate_codes, gate_meta, down_codes, down_meta,
           scalar_weight, workspace, gate_flags, down_flags, capacity):
    """Return BF16 shared output using unchanged IU4 banks for arbitrary M."""
    from ornith_g256 import dense_n32, native
    assert x.ndim == 2 and x.shape[1] == 2048
    assert x.dtype == torch.bfloat16 and x.device.type == 'cuda'
    assert capacity >= 8
    result = torch.empty_like(x)
    for first in range(0, x.shape[0], 8):
        live = min(8, x.shape[0] - first)
        chunk = x[first:first + live].contiguous()
        padded = torch.nn.functional.pad(chunk, (0, 0, 0, 8 - live)) if live < 8 else chunk
        gu = x.new_empty((8, 1024))
        native.check(dense_n32._LAUNCH(
            native.ptr(padded), native.ptr(gate_codes), native.ptr(gate_meta),
            native.ptr(workspace), workspace.numel(), native.ptr(gu),
            native.ptr(gate_flags), 8, capacity, 1024, 2048, 8, 128, 2,
            native.stream(padded)), 'IU4 shared fixed8 gate_up')
        activation = x.new_empty((8, 512))
        torch.ops._C.silu_and_mul(activation, gu)
        down = x.new_empty((8, 2048))
        native.check(dense_n32._LAUNCH(
            native.ptr(activation), native.ptr(down_codes), native.ptr(down_meta),
            native.ptr(workspace), workspace.numel(), native.ptr(down),
            native.ptr(down_flags), 8, capacity, 2048, 512, 8, 128, 2,
            native.stream(activation)), 'IU4 shared fixed8 down')
        # Same BF16 sigmoid/output contract as the unfused Qwen shared MLP.
        # The scalar gate is reduced per row in FP32 and rounded once to BF16.
        result[first:first + live].copy_(torch.sigmoid(gate(chunk, scalar_weight)) * down[:live])
    return result
