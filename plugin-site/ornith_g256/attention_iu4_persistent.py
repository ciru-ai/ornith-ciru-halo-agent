# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru. Isolated optional persistent long-context attention.
"""Page1120 only; canonical BF16 cache owns page identity and prefix lifecycle."""
import ctypes
from functools import lru_cache
import inspect
import linecache
import os
import types

import numpy as np
import torch

_state = None
_installed = False
_library = None
_threshold = 32768


def _compile(source, namespace, suffix):
    name = __file__ + suffix
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    exec(compile(source, name, 'exec'), namespace)


@lru_cache(maxsize=2)
def _bf16_kernels(query_tiles):
    """Original BF16 arithmetic; short requests only, GPU-gated at replay."""
    from . import attention_folded as folded
    native, reduce = folded._kernel(query_tiles)
    source = inspect.getsource(native.fn)
    anchor = '    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor +
        f'    if cur_batch_seq_len >= {_threshold}:\n        return\n')
    namespace = dict(native.fn.__globals__)
    _compile(source, namespace, f'.bf16_{query_tiles}.generated')
    stage1 = namespace[native.fn.__name__]
    namespace = dict(reduce.__globals__)
    stage2 = namespace['_fwd_kernel_stage2']
    source = inspect.getsource(stage2.fn)
    assert source.count(anchor) == 1
    # Padded outputs belong exclusively to the native guarded reducer.
    source = source.replace(anchor, anchor +
        f'    if cur_batch_seq_len >= {_threshold} or cur_batch_seq_len <= 0:\n        return\n')
    _compile(source, namespace, '.bf16_reduce.generated')
    reduce = types.FunctionType(reduce.__code__, namespace, reduce.__name__,
                                reduce.__defaults__, reduce.__closure__)
    return stage1, reduce


class Bank:
    def __init__(self, cache):
        self.cache = cache
        n = cache.shape[0]
        groups = n * 35
        self.packed = [torch.empty(shape, dtype=dtype, device=cache.device)
                       for shape, dtype in [
            ((n, 35, 2, 32, 32), torch.int32),
            ((n, 35, 2, 32), torch.float16),
            ((n, 35, 2, 4, 256), torch.int32),
            ((n, 35, 2, 256), torch.int32)  # FP16 nonDC scale + FP16 normalized V_DC,
        ]]
        self.valid = torch.zeros(groups, dtype=torch.int32, device=cache.device)
        self.busy = torch.zeros_like(self.valid)


class State:
    def __init__(self, runner):
        self.lib = ctypes.CDLL(_library)
        P, I = ctypes.c_void_p, ctypes.c_int
        for name, args in [
            ('update_cache', [P] * 9 + [I, P]),
            ('prepare_decode', [P] * 9 + [I] * 5 + [P]),
            ('cached_attention', [P] * 12 + [I] * 4 + [P]),
            ('tail_reduce', [P] * 9 + [I] * 7 + [P]),
            ('cache_reset', [P] * 3 + [I, P]),
            ('cache_copy', [P] * 7 + [I, P]),
        ]:
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, I
        self.banks = {}
        targets = 0
        for layer in runner.get_model().modules():
            impl = getattr(layer, 'impl', None)
            if (getattr(impl, 'num_heads', None), getattr(impl, 'num_kv_heads', None),
                    getattr(impl, 'head_size', None)) != (16, 2, 256):
                continue
            cache = layer.kv_cache
            if (not isinstance(cache, torch.Tensor) or cache.ndim != 4
                    or cache.shape[1:] != (2, 1120, 512)
                    or cache.dtype != torch.bfloat16
                    or cache.stride() != (1196032, 573440, 512, 1)):
                raise RuntimeError('Persistent IU4 requires bound layer-major target page1120 BF16')
            targets += 1
            if cache.data_ptr() not in self.banks:
                self.banks[cache.data_ptr()] = Bank(cache)
        if targets != 10 or len(self.banks) != 5:
            raise RuntimeError(f'Expected ten target layers aliasing five banks, got {targets}/{len(self.banks)}')
        self.device = next(iter(self.banks.values())).cache.device
        device = self.device
        self.groups = torch.empty(8192, dtype=torch.int32, device=device)
        self.contexts = torch.empty(8, dtype=torch.int32, device=device)
        self.counts = torch.empty(8, dtype=torch.int32, device=device)
        self.owners = torch.empty(64, dtype=torch.int32, device=device)
        self.stats = torch.zeros(3, dtype=torch.int64, device=device)
        self.qp = torch.empty((16, 64, 32), dtype=torch.int32, device=device)
        self.qs = torch.empty((16, 64), dtype=torch.float16, device=device)
        # Both arms use disjoint rows of the same preallocated partial storage.
        self.parts = torch.empty((64, 16, 33, 257), dtype=torch.float32, device=device)
        self.row_seq = torch.empty(64, dtype=torch.int32, device=device)
        self.lse = torch.empty((64, 16), dtype=torch.float32, device=device)
        self.reset_events = self.copy_events = 0

    def call(self, name, tensors, *ints):
        rc = getattr(self.lib, name)(*[t.data_ptr() for t in tensors], *ints,
                                     torch.cuda.current_stream(self.device).cuda_stream)
        if rc:
            raise RuntimeError(f'Persistent IU4 {name} launch failed: {rc}')

    def update(self, cache, slots):
        bank = self.banks.get(cache.data_ptr())
        if bank is None:
            return
        if slots.numel() > 8192 or slots.dtype != torch.int64:
            raise RuntimeError('Unexpected persistent IU4 slot capacity/type')
        self.call('update_cache', [cache, slots, self.groups, bank.busy,
                                   bank.valid, *bank.packed], slots.numel())

    def events(self, scheduler_output):
        from vllm.utils.torch_utils import async_tensor_h2d
        zeros = scheduler_output.new_block_ids_to_zero
        copies = scheduler_output.kv_cache_block_copies
        if zeros:
            ids = async_tensor_h2d(np.asarray(zeros, dtype=np.int64), device=self.device)
            for bank in self.banks.values():
                self.call('cache_reset', [bank.valid, bank.busy, ids], len(zeros))
            self.reset_events += len(zeros)
        if copies:
            pairs_cpu = np.asarray(copies, dtype=np.int64).reshape(-1, 2)
            # Usual CoW destinations are fresh. For dependency chains, retain
            # upstream snapshot-copy semantics using its existing helper.
            if set(pairs_cpu[:, 0]) & set(pairs_cpu[:, 1]):
                from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace
                for bank in self.banks.values():
                    copy_kv_cache_blocks_inplace(
                        [*bank.packed, bank.valid.view(-1, 35), bank.busy.view(-1, 35)],
                        bank.cache.shape[0], copies)
            else:
                pairs = async_tensor_h2d(pairs_cpu, device=self.device)
                for bank in self.banks.values():
                    self.call('cache_copy', [*bank.packed, bank.valid, bank.busy, pairs], len(copies))
            self.copy_events += len(copies)

    def forward(self, query, key_cache, value_cache, output, block_table,
                starts, seq, sm_scale, k_scale, v_scale, max_query_len):
        from vllm.triton_utils import triton
        from .attention_folded import _reduction_lengths
        rows, requests = query.shape[0], seq.numel()
        bank = self.banks[key_cache.data_ptr()]
        tiles = 1 if max_query_len <= 8 else 2
        self.call('prepare_decode', [query, starts, seq, self.contexts,
                  self.counts, self.owners, self.stats, self.qp, self.qs],
                  requests, rows, query.stride(0), query.stride(1), _threshold)
        _reduction_lengths[(rows,)](starts, seq, self.row_seq,
            REQUESTS=requests, REQUEST_BLOCK=triton.next_power_of_2(requests), num_warps=4)
        logits = self.parts[:rows, :, :32, :]
        stage1, reduce = _bf16_kernels(tiles)
        stage1[(requests * tiles, 2, 32)](
            query, key_cache, value_cache, sm_scale, block_table, seq, starts, logits,
            block_table.stride(0), query.stride(0), query.stride(1),
            key_cache.stride(0), key_cache.stride(3), key_cache.stride(1),
            value_cache.stride(0), value_cache.stride(3), value_cache.stride(1),
            logits.stride(0), logits.stride(1), logits.stride(2), k_scale, v_scale,
            kv_group_num=8, q_head_num=16, BLOCK_DMODEL=256, BLOCK_DPE=0,
            BLOCK_DV=256, BLOCK_N=16, BLOCK_H=64, NUM_KV_SPLITS=32,
            PAGE_SIZE=1120, logit_cap=0., Lk=256, Lv=256, IS_MLA=False,
            stride_buf_kds=key_cache.stride(2), stride_buf_kxs=key_cache.stride(4),
            stride_buf_vds=value_cache.stride(2), num_warps=4, num_stages=1,
            waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2)
        reduce(logits, query, output, self.lse[:rows], value_cache.transpose(2, 3), self.row_seq[:rows], 32)
        self.call('cached_attention', [self.qp, self.qs, *bank.packed, self.parts,
                  block_table, self.contexts, self.counts, starts, seq], requests,
                  block_table.stride(0), tiles, _threshold)
        self.call('tail_reduce', [query, bank.cache, block_table, self.contexts,
                  starts, self.owners, seq, self.parts, output], rows,
                  block_table.stride(0), query.stride(0), query.stride(1),
                  output.stride(0), output.stride(1), _threshold)
        return output


def bind(runner):
    global _state
    if _state is not None:
        return 0
    before = torch.cuda.memory_allocated(runner.device)
    _state = State(runner)
    added = torch.cuda.memory_allocated(runner.device) - before
    print('ORNITH_PERSISTENT_IU4_BOUND', len(_state.banks),
          next(iter(_state.banks.values())).cache.shape[0], added, _threshold, flush=True)
    return added


def snapshot():
    if _state is None:
        return None
    return {'bank_count': len(_state.banks), 'gpu_long_short_requests_long_queries': _state.stats.cpu().tolist(),
            'reset_page_events': _state.reset_events, 'copy_page_events': _state.copy_events,
            'threshold': _threshold}


def install():
    global _installed, _library, _threshold
    if _installed:
        return
    _library = os.environ['ORNITH_PERSISTENT_IU4_LIBRARY']
    _threshold = int(os.environ.get('ORNITH_PERSISTENT_IU4_MIN_SEQ', '32768'))
    if not 32768 <= _threshold <= 63000:
        raise ValueError('Persistent IU4 experiment requires a long-only threshold32768..63000')
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from . import attention_folded
    original_update = RocmAttentionImpl.do_kv_cache_update
    original_states = GPUModelRunner._update_states
    original_forward = attention_folded.forward

    def update(self, layer, key, value, kv_cache, slot_mapping):
        result = original_update(self, layer, key, value, kv_cache, slot_mapping)
        if _state is not None:
            _state.update(kv_cache, slot_mapping)
        return result

    def states(self, scheduler_output):
        if _state is not None:
            _state.events(scheduler_output)
        return original_states(self, scheduler_output)

    def forward(query, key_cache, value_cache, output, block_table, query_start_loc,
                seq_lens, sm_scale, k_scale, v_scale, *, max_query_len=8):
        if (_state is None or key_cache.data_ptr() not in _state.banks
                or query.shape[0] > 64 or seq_lens.numel() > 8
                or sm_scale != .0625 or query.dtype != torch.bfloat16
                or output.dtype != torch.bfloat16 or query.stride(2) != 1
                or output.stride(2) != 1 or not 1 <= max_query_len <= 16):
            return original_forward(query, key_cache, value_cache, output,
                block_table, query_start_loc, seq_lens, sm_scale, k_scale, v_scale,
                max_query_len=max_query_len)
        return _state.forward(query, key_cache, value_cache, output, block_table,
            query_start_loc, seq_lens, sm_scale, k_scale, v_scale, max_query_len)

    RocmAttentionImpl.do_kv_cache_update = update
    GPUModelRunner._update_states = states
    attention_folded.forward = forward
    _installed = True
