"""Explicit large-projection policy; routing and small GDN controls stay BF16."""
# Copyright 2026 Ciru.
import re
import torch
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

QUANT_CONFIG = dict(quant_method="ornith_g256", group_size=256,
                    transform_block=128, activation_bits=8)
DENSE_SUFFIXES = (".linear_attn.in_proj_qkvz", ".linear_attn.out_proj",
                  ".self_attn.qkv_proj", ".self_attn.o_proj",
                  ".mlp.shared_expert.gate_up_proj", ".mlp.shared_expert.down_proj")


class OrnithG256Config(QuantizationConfig):
    def __init__(self, config):
        super().__init__()
        if any(config.get(k) != v for k, v in QUANT_CONFIG.items()):
            raise ValueError(f"Expected {QUANT_CONFIG}")
        self.checkpoint_config = dict(config)
        self.activation_bits, self.transform_block = 8, 128

    @classmethod
    def get_name(cls): return "ornith_g256"
    def get_supported_act_dtypes(self): return [torch.bfloat16]
    @classmethod
    def get_min_capability(cls): return 0
    @staticmethod
    def get_config_filenames(): return ["quantize_config.json"]
    @classmethod
    def from_config(cls, config): return cls(config)

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead, VocabParallelEmbedding, UnquantizedEmbeddingMethod,
        )
        from .method import G256MoEMethod, G256LinearMethod, W8HeadMethod
        # MTP weights retain BF16; its temporary head is replaced with the
        # target W8 head by the installed proposer after loading.
        if prefix.startswith('mtp.'):
            if isinstance(layer, RoutedExperts):
                from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            if not re.search(r"layers\.[0-9]+\.mlp\.experts$", prefix):
                raise ValueError(f"Unsupported routed module {prefix}")
            G256MoEMethod.validate_layer_scope(layer)
            return G256MoEMethod(layer.moe_config, self, prefix)
        if isinstance(layer, ParallelLMHead):
            return W8HeadMethod(prefix)
        if isinstance(layer, LinearBase):
            return G256LinearMethod(prefix) if prefix.endswith(DENSE_SUFFIXES) else UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        return None
