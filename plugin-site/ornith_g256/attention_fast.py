"""Optional retained FP32 segmented decode; stock ROCm handles multiple queries."""
# Copyright 2026 Ciru.
import ctypes
from collections import Counter
from types import SimpleNamespace

import torch
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend, RocmAttentionImpl
from vllm.v1.attention.ops.paged_attn import PagedAttention


class PagedLayout(ctypes.Structure):
    _fields_ = [('bytes', ctypes.c_size_t), ('valid_lengths', ctypes.c_size_t),
                ('partial', ctypes.c_size_t), ('segments', ctypes.c_int)]


class PagedStrides(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in
                ('q0', 'q1', 'k0', 'k1', 'k2', 'k3', 'k4', 'v0', 'v1', 'v2', 'v3', 'table0')]


class NativeFastFP32:
    """ABI1 binding only: same library, strides and scratch layout as paged_torch."""
    def __init__(self, library, *, Ccap, Lcap, device):
        if not 1 <= Ccap <= 8 or not 1 <= Lcap <= 8192:
            raise ValueError('FP32 paged decode supports <=8 sequences and <=8192 context')
        self.Ccap, self.Lcap, self.device = Ccap, Lcap, torch.device(device)
        self.library = str(library)
        self.lib = ctypes.CDLL(self.library)
        self.lib.ornith_paged_abi_version.argtypes = []
        self.lib.ornith_paged_abi_version.restype = ctypes.c_uint32
        if self.lib.ornith_paged_abi_version() != 1:
            raise ValueError('Expected retained paged attention ABI1')
        self.lib.ornith_paged_get_layout.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(PagedLayout)]
        self.lib.ornith_paged_get_layout.restype = ctypes.c_int
        self.lib.ornith_paged_launch.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_size_t]
            + [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [PagedStrides, ctypes.c_void_p])
        self.lib.ornith_paged_launch.restype = ctypes.c_int
        layout = PagedLayout()
        if self.lib.ornith_paged_get_layout(Ccap, Lcap, ctypes.byref(layout)):
            raise ValueError('Paged attention workspace query failed')
        self.workspace_bytes = layout.bytes
        self.output_f32_offset = (layout.bytes + 255) // 256 * 256
        self.arena_bytes = self.output_f32_offset + Ccap * 16 * 256 * 4

    def bind_arena(self, arena):
        if (arena.dtype != torch.uint8 or arena.ndim != 1 or not arena.is_contiguous()
                or arena.numel() < self.arena_bytes or arena.data_ptr() % 256
                or arena.device != self.device):
            raise ValueError('Invalid shared attention arena')
        output = arena[self.output_f32_offset:self.arena_bytes].view(torch.float32).view(self.Ccap, 16, 256)
        return SimpleNamespace(workspace=arena[:self.workspace_bytes],
                               outputs=tuple(output[:c] for c in range(self.Ccap + 1)))

    def launch_out(self, query, key, value, table, lengths, buffers, flag, output):
        strides = PagedStrides(*query.stride()[:2], *key.stride(), *value.stride(), table.stride(0))
        stream = torch.cuda.current_stream(query.device)
        status = self.lib.ornith_paged_launch(
            *[tensor.data_ptr() for tensor in (query, key, value, table, lengths, buffers.workspace)],
            buffers.workspace.numel(), buffers.outputs[query.shape[0]].data_ptr(), output.data_ptr(), flag.data_ptr(),
            query.shape[0], self.Ccap, self.Lcap, key.shape[0], strides, stream.cuda_stream)
        if status:
            raise RuntimeError(f'FP32 paged attention launch failed: {status}')


class OrnithG256SelectableAttentionBackend(RocmAttentionBackend):
    @staticmethod
    def get_name(): return 'CUSTOM'

    @staticmethod
    def get_impl_cls():
        settings = get_current_vllm_config().additional_config.get('ornith_g256', {})
        mode = settings.get('attention_mode', 'column')
        if mode == 'stock_rocm':
            return RocmAttentionImpl
        if mode == 'column':
            from .attention import OrnithG256AttentionImpl
            return OrnithG256AttentionImpl
        if mode == 'fast_fp32':
            if not settings.get('attention_library'):
                raise ValueError('fast_fp32 requires additional_config.ornith_g256.attention_library')
            return OrnithG256FastAttentionImpl
        raise ValueError(f'Unknown ornith_g256 attention_mode: {mode}')


class OrnithG256FastAttentionImpl(RocmAttentionImpl):
    implementation = 'ciru.ornith.g256.fast_fp32.abi1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ornith_fast = None
        self._ornith_verify = None
        self.dispatch_counts = Counter()
        self.capture_by_C, self.eager_by_C = Counter(), Counter()
        self.context_bounds = [None, None]

    def static_native_support(self):
        return (self.attn_type == AttentionType.DECODER
            and (self.num_heads, self.num_kv_heads, self.head_size) == (16, 2, 256)
            and self.scale == 0.0625 and self.kv_cache_dtype in ('auto', 'bfloat16')
            and self.alibi_slopes is None and self.sliding_window == (-1, -1)
            and self.logits_soft_cap == 0 and self.sinks is None
            and self.kv_sharing_target_layer_name is None)

    def bind_native(self, backend, buffers, flags, slot, prefix):
        if self._ornith_fast is not None:
            raise RuntimeError('Attention storage already bound')
        if not self.static_native_support():
            raise ValueError('Unsupported Ornith FP32 attention geometry')
        if (flags.dtype != torch.int32 or flags.ndim != 1 or flags.device != backend.device
                or not 0 <= slot < flags.numel()):
            raise ValueError('Invalid attention flag slot')
        self._ornith_fast = (backend, buffers, flags[slot:slot + 1], slot, str(prefix))

    def bind_verify(self, backend, buffers):
        if self._ornith_fast is None or self._ornith_verify is not None:
            raise RuntimeError('Bind verification once after normal attention storage')
        self._ornith_verify = (backend, buffers)

    def _try_verify(self, query, kv_cache, m, output, output_scale, output_block_scale):
        if (self._ornith_verify is None or m is None or not self.static_native_support()
                or m.use_cascade or m.causal is not True or output_scale is not None
                or output_block_scale is not None or not 1 <= m.max_query_len <= 8):
            return False
        backend, buffers = self._ornith_verify
        count, sequences = query.shape[0], m.seq_lens.shape[0]
        if (not 1 <= count <= backend.Ccap or not 1 <= sequences <= 8
                or m.max_seq_len > backend.Lcap or query.shape != (count, 16, 256)
                or output.shape != query.shape or not 0 <= m.num_actual_tokens <= count
                or m.block_table.shape[0] != sequences
                or m.query_start_loc.shape != (sequences + 1,)
                or m.block_table.shape[1] < backend.table_cols
                or query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                or kv_cache.dtype != torch.bfloat16 or m.block_table.dtype != torch.int32
                or m.seq_lens.dtype != torch.int32 or m.query_start_loc.dtype != torch.int32
                or query.stride(2) != 1 or not output.is_contiguous()
                or m.block_table.stride(1) != 1 or not m.seq_lens.is_contiguous()
                or not m.query_start_loc.is_contiguous()):
            return False
        kcache, vcache = PagedAttention.split_kv_cache(kv_cache.transpose(0, 1), 2, 256)
        if kcache.shape[1:] != (2, 32, 1104, 8) or vcache.shape[1:] != (2, 256, 1104):
            return False
        # The inherited stock do_kv_cache_update executes before this attention
        # operation. GPU query lengths mask the newly written future positions.
        backend.launch_out(query, kcache, vcache, m.block_table, m.seq_lens,
                           m.query_start_loc, m.num_actual_tokens, buffers,
                           self._ornith_fast[2], output)
        capturing = torch.cuda.is_current_stream_capturing()
        self.dispatch_counts['capture_verify_calls' if capturing else 'eager_verify_calls'] += 1
        self.dispatch_counts['verify_calls'] += 1
        (self.capture_by_C if capturing else self.eager_by_C)[str(count)] += 1
        return True

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        m = attn_metadata
        if self._try_verify(query, kv_cache, m, output, output_scale, output_block_scale):
            return output
        reason = None
        if m is None:
            reason = 'profile'
        elif not self.static_native_support():
            reason = 'static_feature'
        elif (m.use_cascade or m.causal is not True or output_scale is not None
              or output_block_scale is not None):
            reason = 'metadata_feature'
        elif m.max_query_len != 1:
            reason = 'prefill_or_multiquery'
        elif self._ornith_fast is None:
            raise RuntimeError('FP32 decode reached an unbound target attention layer')
        else:
            backend, buffers, flag, _, _ = self._ornith_fast
            count = m.seq_lens.shape[0]
            if (not 1 <= count <= backend.Ccap or m.max_seq_len > backend.Lcap
                    or query.shape != (count, 16, 256) or output.shape != query.shape
                    or not 0 < m.num_actual_tokens <= count
                    or m.block_table.shape[0] != count or m.query_start_loc.shape != (count + 1,)
                    or m.block_table.shape[1] < (backend.Lcap + 1055) // 1056):
                reason = 'capacity_or_query_mapping'
            elif (query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                    or kv_cache.dtype != torch.bfloat16 or m.block_table.dtype != torch.int32
                    or m.seq_lens.dtype != torch.int32 or m.query_start_loc.dtype != torch.int32):
                reason = 'dtype'
            elif (query.stride(2) != 1 or not output.is_contiguous()
                    or m.block_table.stride(1) != 1 or not m.seq_lens.is_contiguous()
                    or not m.query_start_loc.is_contiguous()):
                reason = 'stride'
            else:
                kcache, vcache = PagedAttention.split_kv_cache(kv_cache.transpose(0, 1), 2, 256)
                if kcache.shape[1:] != (2, 32, 1056, 8) or vcache.shape[1:] != (2, 256, 1056):
                    reason = 'cache_page_shape'
                else:
                    # The existing opaque Attention op owns this current-stream
                    # launch and the preceding stock cache update. No host KV read.
                    backend.launch_out(query, kcache, vcache, m.block_table, m.seq_lens,
                                       buffers, flag, output)
                    capturing = torch.cuda.is_current_stream_capturing()
                    self.dispatch_counts['capture_native_calls' if capturing else 'eager_native_calls'] += 1
                    (self.capture_by_C if capturing else self.eager_by_C)[str(count)] += 1
                    self.dispatch_counts['native_calls'] += 1
                    lo, hi = self.context_bounds
                    self.context_bounds = [m.max_seq_len if lo is None else min(lo, m.max_seq_len),
                                           m.max_seq_len if hi is None else max(hi, m.max_seq_len)]
                    return output
        self.dispatch_counts['fallback_' + reason] += 1
        return super().forward(layer, query, key, value, kv_cache, m, output,
                               output_scale, output_block_scale)

    def inspect_dispatch(self, prefix=None):
        bound = self._ornith_fast
        return dict(implementation=self.implementation, bound=bound is not None,
                    prefix=prefix or (bound[4] if bound else None), flag_slot=bound[3] if bound else None,
                    counts=dict(self.dispatch_counts), capture_by_C=dict(self.capture_by_C),
                    eager_by_C=dict(self.eager_by_C), native_max_seq_len_host_bounds=list(self.context_bounds))
