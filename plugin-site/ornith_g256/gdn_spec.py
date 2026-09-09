# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Ciru modification (2026): zero-initialize output for skipped NULL state rows.
# Wrapper copied from vLLM third_party/flash_linear_attention/ops/
# fused_sigmoid_gating.py; the installed Triton kernel remains unchanged.
"""Defined GDN speculative padding outputs, using the original Triton kernel."""
import torch
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update as upstream_update,
    fused_sigmoid_gating_delta_rule_update_kernel,
)

logger = init_logger(__name__)
_upstream_rocm_core = None


def initialized_rocm_warmup(self, qkvz, ba, z_out, core_attn_out):
    """Define buffers skipped by the metadata-free upstream ROCm warmup."""
    metadata = get_forward_context().attn_metadata
    no_metadata = not isinstance(metadata, dict) or metadata.get(self.prefix) is None
    result = _upstream_rocm_core(self, qkvz, ba, z_out, core_attn_out)
    if no_metadata:
        z_out.zero_()
        core_attn_out.zero_()
        logger.info_once('Ornith GDN metadata-free ROCm warmup initialized z/core outputs')
    return result


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    # NULL_BLOCK_ID causes the original kernel to return without an output
    # store. Valid rows are overwritten exactly as in the upstream wrapper.
    o = q.new_zeros((NK, *v.shape))
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


def install():
    """Rebind only Qwen GDN's imported helper before MTP model construction."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    current = qwen_gdn_linear_attn.fused_sigmoid_gating_delta_rule_update
    if current not in (upstream_update, fused_sigmoid_gating_delta_rule_update):
        raise RuntimeError('Another extension replaced the Qwen GDN update helper')
    qwen_gdn_linear_attn.fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update
    global _upstream_rocm_core
    cls = qwen_gdn_linear_attn.QwenGatedDeltaNetAttention
    current_core = cls._forward_core_rocm
    if _upstream_rocm_core is None:
        _upstream_rocm_core = current_core
    elif current_core not in (_upstream_rocm_core, initialized_rocm_warmup):
        raise RuntimeError('Another extension replaced the Qwen GDN ROCm core')
    cls._forward_core_rocm = initialized_rocm_warmup
