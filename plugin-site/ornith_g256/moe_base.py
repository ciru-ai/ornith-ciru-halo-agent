"""Compact loader and explicitly configured grouped vLLM execution boundary."""
# Copyright 2026 Ciru.
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase


class NativeImplementationUnavailable(RuntimeError):
    pass


class OrnithMoEMethodBase(FusedMoEMethodBase):
    """Unchanged routing/geometry contract inherited by the G256 method."""
    def __init__(self, moe, quant_config, prefix):
        super().__init__(moe)
        self.quant_config, self.prefix = quant_config, prefix
        self._grouped_runtime = None
        self._grouped_layer_index = None
        if (moe.num_experts, moe.num_local_experts, moe.num_logical_experts,
                moe.experts_per_token, moe.hidden_dim, moe.intermediate_size) != (256, 256, 256, 8, 2048, 512):
            raise ValueError("ornith_tilebank requires E256/top8/H2048/I512 without shared/redundant expert slots")
        if moe.in_dtype != torch.bfloat16 or moe.activation != MoEActivation.SILU:
            raise ValueError("ornith_tilebank requires BF16 inputs and ordinary SiLU gating")
        if moe.has_bias or moe.is_lora_enabled or any(
                getattr(moe, name, None) is not None for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta")):
            raise ValueError("Bias, LoRA and modified SwiGLU are outside the initial Ornith contract")
        self._validate_parallel(moe.moe_parallel_config)
        if moe.aiter_fmoe_shared_expert_enabled or moe.rocm_aiter_fmoe_enabled:
            raise ValueError("Disable AITER fused MoE/shared experts for Ornith IU4 bringup")
        if moe.defer_moe_finalize or any(getattr(moe, name, None) is not None
                for name in ('activation_situ_beta', 'activation_situ_linear_beta')):
            raise ValueError('Deferred finalize and modified activations are unsupported')

    @staticmethod
    def _validate_parallel(parallel):
        if any(getattr(parallel, key) != 1 for key in ("tp_size", "ep_size", "dp_size", "pcp_size", "sp_size")):
            raise ValueError("Initial ornith_tilebank adapter supports one rank, no sequence parallelism")
        if parallel.use_ep or parallel.enable_eplb:
            raise ValueError("Initial ornith_tilebank adapter does not support EP/EPLB")

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition,
                           act_dtype, moe_parallel_config):
        self._validate_parallel(moe_parallel_config)
        if (hidden_size, intermediate_size_per_partition, act_dtype) != (2048, 512, torch.bfloat16):
            raise ValueError("Unexpected Ornith shape/dtype; implicit padding is forbidden")
        return hidden_size, intermediate_size_per_partition

    @property
    def skip_forward_padding(self):
        return True

    @property
    def topk_indices_dtype(self):
        return torch.int32

    @staticmethod
    def validate_layer_scope(layer):
        if (layer.apply_router_weight_on_input or layer.use_grouped_topk
                or layer.custom_routing_function is not None
                or layer.num_expert_group is not None or layer.topk_group is not None
                or layer.e_score_correction_bias is not None
                or layer.scoring_func != 'softmax' or not layer.renormalize
                or layer.routed_scaling_factor != 1.0):
            raise ValueError('Ornith requires unchanged ordinary renormalized top8 routing')

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        raise NativeImplementationUnavailable("Ornith does not implement monolithic router execution")
