"""Exact greedy DFlash2 path walk; candidate logits and edge scores unchanged."""
import torch
import triton
import triton.language as tl


@triton.jit
def _walk(S, I, O, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
          S3: tl.constexpr, I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
          L: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, K)
    predecessor = tl.full((), 0, tl.int32)
    for step in tl.static_range(L):
        values = tl.load(S + row*S0 + step*S1 + predecessor*S2 + col*S3).to(tl.float32)
        # Torch argmax selects the first NaN, or the first maximum otherwise.
        nan_index = tl.min(tl.where(values != values, col, K), 0)
        maximum = tl.max(tl.where(values != values, -float('inf'), values), 0)
        max_index = tl.min(tl.where(values == maximum, col, K), 0)
        selected = tl.where(nan_index < K, nan_index, max_index)
        token = tl.load(I + row*I0 + step*I1 + selected*I2)
        tl.store(O + row*L + step, token)
        predecessor = selected


def greedy_path(scores, candidate_ids):
    b, length, k = candidate_ids.shape
    if scores.shape != (b, length, k, k) or k != 16 or not 1 <= length <= 15:
        raise ValueError('Unqualified DFlash2 score/ID shape')
    if scores.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('Unqualified score dtype')
    if candidate_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Unqualified candidate-ID dtype')
    if not scores.is_cuda or candidate_ids.device != scores.device:
        raise ValueError('DFlash path requires matching GPU tensors')
    output = torch.empty((b * length,), device=candidate_ids.device, dtype=candidate_ids.dtype)
    if b:
        _walk[(b,)](scores, candidate_ids, output, *scores.stride(),
                    *candidate_ids.stride(), length, k, num_warps=1)
    return output
