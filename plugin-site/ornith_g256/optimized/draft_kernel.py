"""Single-pass convolution + residual RMSNorm; no scratch reread."""
import triton
import triton.language as tl

@triton.jit
def fused_norm_register(residual, base, coefficients, hidden, query_mask, weight,
                        out_residual, out_norm, rows: tl.constexpr,
                        RETURN_RESIDUAL: tl.constexpr, XBLOCK: tl.constexpr = 2):
    row = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    col = tl.arange(0, 2048)[None, :]
    valid = row < rows
    mask = tl.load(query_mask)
    b0 = tl.load(base + 4096 + col).to(tl.float32)
    b1 = tl.load(base + 6144 + col).to(tl.float32)
    c0 = tl.load(coefficients + 256 + 512 * row + col // 16, valid, 0).to(tl.float32)
    c1 = tl.load(coefficients + 384 + 512 * row + col // 16, valid, 0).to(tl.float32)
    x = tl.load(hidden + 2048 * row + col, valid, 0).to(tl.float32)
    previous = tl.load(hidden + 2048 * (row - 1) + col, valid & (row >= 1), 0).to(tl.float32)
    old = tl.load(residual + 2048 * row + col, valid, 0)
    w = tl.load(weight + col).to(tl.float32)
    coefficient0 = b0 + c0
    coefficient1 = b1 + c1
    term0 = coefficient0 * x
    term1 = coefficient1 * previous
    term1 = term1 * ((row & mask) >= 1).to(tl.float32)
    convolved = term0 + term1
    updated = convolved + old
    square_sum = tl.sum(updated * updated, axis=1)[:, None]
    inv = tl.rsqrt(square_sum / 2048.0 + 1e-6)
    normalized = (updated * inv) * w
    tl.store(residual + 2048 * row + col, updated, valid)
    if RETURN_RESIDUAL:
        tl.store(out_residual + 2048 * row + col, updated, valid)
    tl.store(out_norm + 2048 * row + col, normalized, valid)

def launch(kind, inputs, stream):
    import torch
    assert stream == torch.cuda.current_stream().cuda_stream
    if kind == 'middle':
        residual, base, coef, hidden, mask, weight, out_residual, out_norm, rows, width = inputs
    else:
        residual, base, coef, hidden, mask, weight, out_norm, rows, width = inputs
        out_residual = out_norm  # ignored by the final-layer specialization
    assert width == 2048
    assert residual.dtype == torch.float32 and tuple(residual.shape) == (rows, 2048)
    assert all(t.is_contiguous() for t in [residual, base, coef, hidden, mask, weight, out_residual, out_norm])
    assert tuple(coef.shape) == (rows, 512) and tuple(base.shape) == (2, 2, 2048)
    fused_norm_register[(triton.cdiv(rows, 2),)](residual, base, coef, hidden, mask, weight,
        out_residual, out_norm, rows, kind == 'middle', XBLOCK=2, num_warps=8,
        num_stages=1, enable_fp_fusion=False)
