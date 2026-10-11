"""BF16 operands with rowwise FP32 reductions, matching the Ultra repairs."""
import torch
from vllm.triton_utils import triton, tl


@triton.jit
def _ba_rows(X, W, Y, SX: tl.constexpr, SY: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 2048)
    x = tl.load(X + row * SX + k).to(tl.float32)
    w = tl.load(W + n[:, None] * 2048 + k[None, :]).to(tl.float32)
    value = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + row * SY + n, value)


@triton.jit
def _router_rows(X, W, Y, SX: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 2048)
    x = tl.load(X + row * SX + k).to(tl.float32)
    w = tl.load(W + n[:, None] * 2048 + k[None, :]).to(tl.float32)
    value = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + row * 256 + n, value)


def ba_into(x, weight, out):
    assert x.dtype == weight.dtype == out.dtype == torch.bfloat16
    assert x.ndim == 2 and x.shape[1] == 2048 and weight.shape == (64, 2048)
    assert out.shape == (x.shape[0], 64) and x.stride(1) == out.stride(1) == 1
    assert weight.is_contiguous()
    return _ba_rows[(x.shape[0], 16)](x, weight, out, x.stride(0), out.stride(0), 4, num_warps=4)


def router(x, weight):
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.ndim == 2 and x.shape[1] == 2048 and weight.shape == (256, 2048)
    assert x.stride(1) == 1 and weight.is_contiguous()
    out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
    kernel = _router_rows[(x.shape[0], 64)](x, weight, out, x.stride(0), 4, num_warps=4)
    return out, kernel
