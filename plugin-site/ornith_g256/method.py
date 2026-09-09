"""Direct packed loads and fresh-output linear/MoE implementations."""
# Copyright 2026 Ciru.
import torch
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from .moe_base import OrnithMoEMethodBase
from .loader import install_tilebank_loader, require_complete_tilebank_load


def register_packed(layer, specs):
    layer._g256_loaded = set()
    layer._g256_ready = False
    for name, shape, dtype in specs:
        param = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        layer.register_parameter(name, param)

        def load(destination, value, *shard_args, name=name, param=param):
            if layer._g256_ready or name in layer._g256_loaded or destination is not param:
                raise ValueError(f"Duplicate or late packed load: {name}")
            if value.shape != param.shape or value.dtype != param.dtype:
                raise ValueError(f"Packed shape/dtype mismatch: {name}: {value.shape}/{value.dtype}")
            # The fused qkvz checkpoint alias acquires a vLLM shard_id; its
            # payload already contains all four projections, so no slicing.
            with torch.no_grad(): param.copy_(value)
            layer._g256_loaded.add(name)
        param.weight_loader = load


class G256LinearMethod(LinearMethodBase):
    fields = ("g256_codes", "g256_metadata")

    def __init__(self, prefix):
        self.prefix, self.runtime, self.slot = prefix, None, None

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        self.k, self.n = input_size, output_size
        if (input_size_per_partition != input_size or sum(output_partition_sizes) != output_size
                or getattr(layer, 'tp_size', 1) != 1 or params_dtype != torch.bfloat16
                or getattr(layer, 'has_bias', False) or not 0 < self.n <= 12288
                or self.n % 16 or not 0 < self.k <= 8192 or self.k % 256):
            raise ValueError(f"Unsupported G256 linear geometry: {self.prefix}: N{self.n}/K{self.k}")
        register_packed(layer, [(self.fields[0], (self.n//16, self.k//256, 32, 16), torch.uint32),
                                (self.fields[1], (self.n//16, self.k//256, 16), torch.uint32)])

    def process_weights_after_loading(self, layer):
        if layer._g256_loaded != set(self.fields):
            raise ValueError(f"Incomplete packed weights: {self.prefix}")
        layer._g256_ready = True

    def apply(self, layer, x, bias=None):
        if self.runtime is None or not layer._g256_ready or bias is not None:
            raise RuntimeError("G256 requires loaded weights and a bound worker")
        shape = x.shape[:-1]
        x = x.reshape(-1, self.k).contiguous()
        out = torch.empty((x.shape[0], self.n), dtype=x.dtype, device=x.device)
        from .native import dense_out
        dense_out(x, layer.g256_codes, layer.g256_metadata, self.runtime.workspace,
                  out, self.runtime.slots[self.slot], self.runtime.capacity,
                  self.n, self.k, self.runtime.dense_geometry, self.runtime.dense_a8_max_rows)
        return out.view(*shape, self.n)


class W8HeadMethod(G256LinearMethod):
    fields = ("head_tilebank_codes", "head_scales")

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        self.k, self.n = input_size, output_size
        if (input_size_per_partition, input_size, output_size, params_dtype,
                getattr(layer, 'tp_size', 1)) != (2048, 2048, 248320, torch.bfloat16, 1):
            raise ValueError("W8 head requires the full TP1 Ornith vocabulary")
        register_packed(layer, [(self.fields[0], (15520, 128, 4, 16), torch.uint32),
                                (self.fields[1], (248320, 16), torch.float16)])

    def apply(self, layer, x, bias=None):
        if self.runtime is None or not layer._g256_ready or bias is not None:
            raise RuntimeError("W8 head requires loaded weights and a bound worker")
        x = x.reshape(-1, self.k).contiguous()
        out = torch.empty((x.shape[0], self.n), dtype=x.dtype, device=x.device)
        from .native import head_out
        # Startup dummy logits can have more rows than normal C1/C8 sampling.
        for start in range(0, x.shape[0], 64):
            head_out(x[start:start+64], layer.head_tilebank_codes, layer.head_scales,
                     self.runtime.workspace, out[start:start+64], self.runtime.slots[self.slot],
                     self.runtime.head_geometry)
        return out


class G256DispatchConfig(FusedMoEQuantConfig):
    def __init__(self):
        super().__init__(
            _a1=FusedMoEQuantDesc(dtype='ornith_s8_g256', shape=GroupShape(1, 256)),
            _a2=FusedMoEQuantDesc(dtype='ornith_s8_g256', shape=GroupShape(1, 256)),
            _w1=FusedMoEQuantDesc(dtype='ornith_affine_u4_g256', shape=GroupShape(1, 256)),
            _w2=FusedMoEQuantDesc(dtype='ornith_affine_u4_g256', shape=GroupShape(1, 256)),
            is_scale_swizzled=False)

    @property
    def ocp_mx_scheme(self): return None
    def config_name(self, dtype):
        raise RuntimeError("G256 uses its own native dispatch")


class G256MoEMethod(OrnithMoEMethodBase):
    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        if (num_experts, hidden_size, intermediate_size_per_partition, params_dtype) != (256, 2048, 512, torch.bfloat16):
            raise ValueError("G256 routed weights require E256/H2048/I512 BF16")
        self.runtime, self.slot = None, None
        layer._ornith_weights_ready = False
        for proj, n, k in (("w13", 1024, 2048), ("w2", 2048, 512)):
            for suffix, shape in (("codes", (256, n//16, k//256, 32, 16)),
                                  ("metadata", (256, n//16, k//256, 16))):
                layer.register_parameter(f"{proj}_tilebank_{suffix}",
                    torch.nn.Parameter(torch.empty(shape, dtype=torch.uint32), requires_grad=False))
        install_tilebank_loader(layer)

    def process_weights_after_loading(self, layer):
        require_complete_tilebank_load(layer)
        layer._ornith_weights_ready = True

    def prepare_n32_weights(self, layer, *, replace=False):
        """Reorder at startup, optionally replacing the loaded N16 storage."""
        if not layer._ornith_weights_ready or self.runtime is not None:
            raise RuntimeError('N32 shadows require loaded, unbound routed weights')
        if getattr(layer, '_ornith_storage_n32', False):
            raise RuntimeError('N32 expert storage is prepared once')
        total = 0
        with torch.no_grad():
            for proj in ('w13', 'w2'):
                for suffix in ('codes', 'metadata'):
                    name = f'{proj}_n32_{suffix}'
                    if hasattr(layer, name):
                        raise RuntimeError('N32 routed shadows are prepared once')
                    parent = getattr(layer, f'{proj}_tilebank_{suffix}')
                    e, nt, groups = parent.shape[:3]
                    if (parent.dtype != torch.uint32 or parent.device.type != 'cuda'
                            or not parent.is_contiguous() or nt % 2):
                        raise ValueError('N32 shadows require contiguous GPU U32 N16 banks')
                    if suffix == 'codes':
                        shadow = parent.reshape(e, nt//2, 2, groups, 32, 16).permute(
                            0, 1, 3, 4, 2, 5).contiguous().reshape(e, nt//2, groups, 32, 32)
                    else:
                        shadow = parent.reshape(e, nt//2, 2, groups, 16).permute(
                            0, 1, 3, 2, 4).contiguous().reshape(e, nt//2, groups, 32)
                    if replace:
                        # Preserve the registered parameter object/loader. Its
                        # payload changes layout once, before binding/capture.
                        # The source checkpoint remains in its load format.
                        parent.data = shadow
                    else:
                        layer.register_buffer(name, shadow, persistent=False)
                    total += shadow.numel()*shadow.element_size()
        if replace:
            layer._ornith_storage_n32 = True
        return total

    def get_fused_moe_quant_config(self, layer):
        if self.moe_quant_config is None:
            self.moe_quant_config = G256DispatchConfig()
        return self.moe_quant_config

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        if self.runtime is None or not layer._ornith_weights_ready:
            raise RuntimeError("G256 routed weights require a bound worker")
        from .native import routed_out, routed_out_n32
        out = torch.empty_like(x)
        if self.runtime.routed_decode_n32:
            routed_out_n32(x, topk_weights, topk_ids, layer.w13_tilebank_codes,
                           layer.w13_tilebank_metadata, layer.w2_tilebank_codes,
                           layer.w2_tilebank_metadata, layer.w13_n32_codes,
                           layer.w13_n32_metadata, layer.w2_n32_codes,
                           layer.w2_n32_metadata, self.runtime.workspace, out,
                           self.runtime.slots[self.slot], self.runtime.capacity,
                           self.runtime.routed_n32_max_rows)
            return out
        routed_out(x, topk_weights, topk_ids, layer.w13_tilebank_codes,
                   layer.w13_tilebank_metadata, layer.w2_tilebank_codes,
                   layer.w2_tilebank_metadata, self.runtime.workspace, out,
                   self.runtime.slots[self.slot], self.runtime.capacity)
        return out
