"""Single-program softmax top-k routing for decode-sized E256/top8 batches."""
import torch
import triton
import triton.language as tl

_INSTALLED = False


@triton.jit
def _topk_softmax(logits_ptr, weights_ptr, ids_ptr, stride, E: tl.constexpr, TOPK: tl.constexpr, RENORM: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, E)
    x = tl.load(logits_ptr + row * stride + offs).to(tl.float32)
    p = tl.exp(x - tl.max(x, axis=0))
    p = p / tl.sum(p, axis=0)
    slots = tl.arange(0, TOPK)
    chosen_w = tl.zeros([TOPK], dtype=tl.float32)
    chosen_i = tl.zeros([TOPK], dtype=tl.int32)
    for k in tl.static_range(TOPK):
        best = tl.max(p, axis=0)
        index = tl.min(tl.where(p == best, offs, E), axis=0)
        chosen_w = tl.where(slots == k, best, chosen_w)
        chosen_i = tl.where(slots == k, index, chosen_i)
        p = tl.where(offs == index, -1.0, p)
    if RENORM:
        chosen_w = chosen_w / tl.sum(chosen_w, axis=0)
    tl.store(weights_ptr + row * TOPK + slots, chosen_w)
    tl.store(ids_ptr + row * TOPK + slots, chosen_i)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.layers.fused_moe.router import fused_topk_router
    original = fused_topk_router.dispatch_topk_softmax_func

    def dispatch(use_rocm_aiter=False):
        fallback = original(use_rocm_aiter=use_rocm_aiter)

        def topk(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize):
            rows, experts = gating_output.shape
            if (rows > 16 or experts != 256 or topk_weights.shape[1] != 8 or topk_ids.dtype != torch.int32
                    or not gating_output.is_contiguous() or not topk_weights.is_contiguous()
                    or not topk_ids.is_contiguous()):
                return fallback(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize)
            _topk_softmax[(rows,)](gating_output, topk_weights, topk_ids, gating_output.stride(0),
                                   E=experts, TOPK=8, RENORM=bool(renormalize), num_warps=4)
            return topk_weights, topk_ids
        return topk

    fused_topk_router.dispatch_topk_softmax_func = dispatch
    _INSTALLED = True
