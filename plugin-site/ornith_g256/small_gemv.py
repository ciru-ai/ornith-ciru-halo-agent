"""Row-parallel Triton GEMV for few-row BF16 projections and INT8 DFlash weights."""
import torch
import triton
import triton.language as tl

_INSTALLED = False
_INT8 = {}


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, out_ptr, M, N, K: tl.constexpr, BN: tl.constexpr):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, K)
    cmask = cols < N
    w = tl.load(w_ptr + cols[:, None] * K + ks[None, :], mask=cmask[:, None], other=0.0).to(tl.float32)
    for m in range(M):
        x = tl.load(x_ptr + m * K + ks).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        tl.store(out_ptr + m * N + cols, acc.to(out_ptr.dtype.element_ty), mask=cmask)


@triton.jit
def _int8_gemv_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K: tl.constexpr, KP: tl.constexpr, BN: tl.constexpr):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, KP)
    cmask = cols < N
    kmask = ks < K
    w = tl.load(w_ptr + cols[:, None] * K + ks[None, :], mask=cmask[:, None] & kmask[None, :], other=0).to(tl.float32)
    scale = tl.load(s_ptr + cols, mask=cmask, other=0.0)
    for m in range(M):
        x = tl.load(x_ptr + m * K + ks, mask=kmask, other=0.0).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1) * scale
        tl.store(out_ptr + m * N + cols, acc.to(out_ptr.dtype.element_ty), mask=cmask)


def _int8(weight):
    entry = _INT8.get(weight.data_ptr())
    if entry is None:
        w = weight.float()
        scale = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        entry = (torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8).contiguous(), scale.contiguous())
        _INT8[weight.data_ptr()] = entry
    return entry


def _route(x, weight, bias, draft):
    rows = x.numel() // x.shape[-1]
    n, k = weight.shape
    if (bias is not None or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not weight.is_contiguous() or k % 256 or k > 16384):
        return None
    if draft and rows <= 4 and n * k >= (1 << 22):
        return 'int8'
    if k & (k - 1) == 0 and (rows <= 4 or (rows <= 8 and n <= 1024 and k <= 4096)):
        return 'bf16'
    return None


@torch.library.custom_op('ornith_g256::small_gemm', mutates_args=(), device_types='cuda')
def small_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, draft: bool) -> torch.Tensor:
    route = _route(x, weight, bias, draft)
    if route is None:
        return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)
    n, k = weight.shape
    x2 = x.reshape(-1, k).contiguous()
    out = torch.empty((x2.shape[0], n), dtype=x.dtype, device=x.device)
    if route == 'int8':
        q, scale = _int8(weight)
        bn = 1 if k > 4096 else 2
        _int8_gemv_kernel[(triton.cdiv(n, bn),)](x2, q, scale, out, x2.shape[0], n, K=k,
                                                 KP=triton.next_power_of_2(k), BN=bn, num_warps=4)
    else:
        _gemv_kernel[(triton.cdiv(n, 2),)](x2, weight, out, x2.shape[0], n, K=k, BN=2, num_warps=4)
    return out.reshape(*x.shape[:-1], n)


@small_gemm.register_fake
def _small_gemm_fake(x, weight, bias, draft):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _gemm(layer, x, weight, bias=None):
    return torch.ops.ornith_g256.small_gemm(x, weight, bias, bool(getattr(weight, '_ornith_draft', False)))


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.layers import linear
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from vllm import _custom_ops as ops
    original_dispatch = linear.dispatch_unquantized_gemm

    def dispatch(*args, **kwargs):
        from vllm.platforms import current_platform
        return _gemm if current_platform.is_rocm() else original_dispatch(*args, **kwargs)

    def project_context_kv(self, context_states, num_ctx, num_layers, num_kv_heads, head_dim):
        normed = torch.empty_like(context_states)
        ops.rms_norm(normed, context_states, self._hidden_norm_weight, self._rms_norm_eps)
        flat = torch.ops.ornith_g256.small_gemm(normed, self._fused_kv_weight, self._fused_kv_bias, True)
        kv = flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim).permute(2, 1, 0, 3, 4).contiguous()
        return kv[0], kv[1]

    original_load = DFlashProposer.load_model

    def load_model(self, target_model):
        result = original_load(self, target_model)
        target = {id(p) for p in target_model.parameters()}
        for parameter in self.model.parameters():
            if id(parameter) not in target:
                parameter._ornith_draft = True
        return result

    linear.dispatch_unquantized_gemm = dispatch
    DFlashQwen3Model._project_context_kv = project_context_kv
    DFlashProposer.load_model = load_model
    _INSTALLED = True
