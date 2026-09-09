"""One serialized native arena, bound before memory profiling and capture."""
# Copyright 2026 Ciru.
from types import SimpleNamespace
import torch
import vllm.envs as envs
from vllm.logger import init_logger
from .method import G256LinearMethod, G256MoEMethod, W8HeadMethod
from .native import configure

logger = init_logger('vllm.ornith_g256.runtime')


class Runtime:
    def __init__(self, model, settings, capacity, device, *, max_model_len=4096, max_num_seqs=8):
        if not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            raise ValueError("Set VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 for the shared arena")
        self.backend = SimpleNamespace(device=torch.device(device))
        self.capacity = capacity
        self.routed_decode_n32 = settings.get('routed_decode_n32', False)
        self.routed_storage_n32 = settings.get('routed_storage_n32', False)
        self.routed_n32_max_rows = settings.get('routed_n32_max_rows', 8)
        self.routed_n32_bytes = 0
        self.dense_geometry = settings.get('dense_geometry', 2)
        self.dense_a8_max_rows = settings.get('dense_a8_max_rows', 32)
        if self.dense_a8_max_rows not in (32, 64):
            raise ValueError('G256 dense A8 crossover must be32 or64 rows')
        self.head_geometry = settings.get('head_geometry', 2)
        if self.dense_geometry not in (0, 1, 2) or self.head_geometry not in (0, 1, 2):
            raise ValueError("Native geometry must be 0, 1, or 2")
        layers = [(name, layer) for name, layer in model.named_modules()
                  if isinstance(getattr(layer, 'quant_method', None), (G256LinearMethod, G256MoEMethod))]
        routed = [layer for _, layer in layers if isinstance(layer.quant_method, G256MoEMethod)]
        heads = [layer for _, layer in layers if isinstance(layer.quant_method, W8HeadMethod)]
        dense = [layer for _, layer in layers if type(layer.quant_method) is G256LinearMethod]
        if (len(routed), len(dense), len(heads)) != (40, 160, 1):
            raise ValueError(f"Expected 40 routed/160 dense/1 head, got {len(routed)}/{len(dense)}/{len(heads)}")
        shapes = sorted({(layer.quant_method.n, layer.quant_method.k) for layer in dense})
        size = configure(settings, capacity, shapes)
        if self.routed_storage_n32:
            repacked = sum(layer.quant_method.prepare_n32_weights(layer, replace=True)
                           for layer in routed)
            logger.info('Ornith sole N32 expert storage ready: %d layers, %d bytes repacked; '
                        'no N16 shadows retained', len(routed), repacked)
        if self.routed_decode_n32:
            for layer in routed:
                self.routed_n32_bytes += layer.quant_method.prepare_n32_weights(layer)
            logger.info('Ornith N32 decode shadows ready: %d layers, %d bytes; T<=%d',
                        len(routed), self.routed_n32_bytes, self.routed_n32_max_rows)
        attention = []
        self.attention_backend = self.attention_buffers = None
        self.attention_verify_backend = self.attention_verify_buffers = None
        if settings.get('attention_mode', 'column') == 'fast_fp32':
            from .attention_fast import NativeFastFP32, OrnithG256FastAttentionImpl
            attention = [(name, layer) for name, layer in model.named_modules()
                         if isinstance(getattr(layer, 'impl', None), OrnithG256FastAttentionImpl)]
            if len(attention) != 10:
                raise ValueError(f'Expected ten target fast-attention layers, got {len(attention)}')
            self.attention_backend = NativeFastFP32(settings['attention_library'],
                Ccap=max_num_seqs, Lcap=min(max_model_len, 8192), device=device)
            size = max(size, self.attention_backend.arena_bytes)
            if settings.get('attention_verify_library'):
                from .attention_verify import NativeVerifyFP32
                self.attention_verify_backend = NativeVerifyFP32(settings['attention_verify_library'],
                    Ccap=max_num_seqs * 8, Lcap=min(max_model_len, 8192), device=device)
                size = max(size, self.attention_verify_backend.arena_bytes)
        self.workspace = torch.empty(size, dtype=torch.uint8, device=device)
        self.error_flags = torch.zeros(len(layers)+len(attention), dtype=torch.int32, device=device)
        self.slots = tuple(self.error_flags[i:i+1] for i in range(len(layers)+len(attention)))
        self.layer_prefixes = tuple(name for name, _ in layers+attention)
        self.layers = tuple(layer for _, layer in layers)
        for slot, layer in enumerate(self.layers):
            method = layer.quant_method
            if method.runtime is not None: raise RuntimeError("Native layer already bound")
            method.runtime, method.slot = self, slot
        if self.attention_backend is not None:
            self.attention_buffers = self.attention_backend.bind_arena(self.workspace)
            if self.attention_verify_backend is not None:
                self.attention_verify_buffers = self.attention_verify_backend.bind_arena(self.workspace)
            for slot, (name, layer) in enumerate(attention, start=len(layers)):
                layer.impl.bind_native(self.attention_backend, self.attention_buffers,
                                       self.error_flags, slot, name)
                if self.attention_verify_backend is not None:
                    layer.impl.bind_verify(self.attention_verify_backend, self.attention_verify_buffers)
        self.shapes = shapes

    def validate_identity(self): pass

    def reset_error_flags(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Reset native error flags outside graph capture")
        self.error_flags.zero_()

    def check_error_flags(self):
        flags = self.error_flags.cpu().tolist()
        if any(flags):
            raise RuntimeError(str({name: code for name, code in zip(self.layer_prefixes, flags) if code}))
        return flags
