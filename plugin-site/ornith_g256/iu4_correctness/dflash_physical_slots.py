"""Source-bound physical DFlash slots; RoPE, allocation and sampling unchanged.

Adapted from Ciru's Q8 physical-slot repair. Target M-RoPE is a rotary position,
not an index into the draft group's physical block table. Recompute only the
two slot buffers after the retained input/boundary/per-group hooks execute.
The same input hook maintains context during K0 and speculative re-entry.
"""
# Copyright 2026 Ciru. Native compatibility sources are Apache-2.0.
from functools import wraps
import hashlib
import inspect
import json
from numbers import Integral
import os
from pathlib import Path
from types import MethodType

import torch
from vllm.triton_utils import tl, triton


NATIVE_SOURCE_SHA256 = {
    'dflash': '56b817daecb3740cad899a3bc1ebc481def465457d5e1a7a516e0535522e9e7f',
    'utils': 'edbd1a85a9c372ac3b565453bbe6e37e8b49810b76976f4994dab8970b904af3',
}
HOOK_SOURCE_SHA256 = {
    'dflash_spec': 'c948b99eab22d2af5c5fc0ca9f764a0855a3d86aca0609362bf5474a9beff1b3',
    'dflash_conv_boundary': 'a5f46458194cade5ddcbfbb51af30c662d12b9086fc459ecbeb17463339febb1',
}
_STATE = '_ciru_dflash_physical_slots_state'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@triton.jit
def _physical_slots(Starts, Seq, Table, Rejected, ContextSlots, QuerySlots,
                    TABLE_STRIDE: tl.constexpr, COLS: tl.constexpr,
                    PAGE: tl.constexpr, WIDTH: tl.constexpr,
                    context_size, HAS_REJECTED: tl.constexpr,
                    BLOCK: tl.constexpr):
    request = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    count = end - begin
    seq = tl.load(Seq + request)
    rejected = 0
    if HAS_REJECTED:
        rejected = tl.load(Rejected + request)
    context = j < count
    query = (j >= count) & (j < count + WIDTH)
    query_index = j - count
    position = tl.where(context, seq - count + j, seq - rejected + query_index)
    # Empty/padded rows and out-of-range positions must not retain stale slots.
    valid = (context | query) & (count > 0) & (position >= 0) & (position < COLS * PAGE)
    page = tl.load(Table + request * TABLE_STRIDE + position // PAGE,
                   valid, other=0).to(tl.int64)
    # Native sliding-window allocation replaces retired/unallocated pages with
    # null block zero. Never turn these placeholders into context/query writes.
    slot = tl.where(valid & (page > 0), page * PAGE + position % PAGE, -1)
    context_index = begin + j
    tl.store(ContextSlots + context_index, slot,
             context & (context_index >= 0) & (context_index < context_size))
    tl.store(QuerySlots + request * WIDTH + query_index, slot, query)


def _source_contract(dflash, utils, slot_hook, boundary):
    modules = {'dflash': dflash, 'utils': utils,
               'dflash_spec': slot_hook, 'dflash_conv_boundary': boundary}
    expected = {**NATIVE_SOURCE_SHA256, **HOOK_SOURCE_SHA256}
    records = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        actual = _sha(path)
        if actual != expected[name]:
            raise RuntimeError('DFlash physical-slot source differs: '+name)
        records[name] = {'path': str(path), 'sha256': actual}
    native = slot_hook._upstream_set_inputs_first_pass
    if (native.__module__ != 'vllm.v1.spec_decode.dflash'
            or native.__qualname__ != 'DFlashProposer.set_inputs_first_pass'
            or Path(inspect.getsourcefile(native)).resolve() != Path(dflash.__file__).resolve()):
        raise RuntimeError('DFlash physical slots require the retained native input method')
    current = dflash.DFlashProposer.set_inputs_first_pass
    if (not boundary._INSTALLED
            or Path(inspect.getsourcefile(current)).resolve() != Path(boundary.__file__).resolve()
            or inspect.getclosurevars(current).nonlocals.get('original_inputs') is not slot_hook.set_inputs_first_pass
            or not isinstance(dflash.copy_and_expand_dflash_inputs_kernel, boundary._InputKernelProxy)
            or dflash.copy_and_expand_dflash_inputs_kernel.original is not utils.copy_and_expand_dflash_inputs_kernel):
        raise RuntimeError('DFlash physical slots require the retained boundary and per-group hooks')
    return records


def _validate_backend(instance, allow_pending=False):
    groups = instance.draft_attn_groups
    page, gid = instance.block_size, instance.kv_cache_gid
    # SpecDecodeBaseProposer initializes this geometry only after load_model.
    if allow_pending and page == -1 and gid == -1 and not groups:
        return False
    if (page != 560 or not isinstance(gid, Integral) or isinstance(gid, bool)
            or gid < 0 or not groups
            or any(g.kv_cache_group_id != gid for g in groups)):
        raise RuntimeError('DFlash physical slots require initialized page560 and a single native draft group')
    return True


def _validate_inputs(instance, target_token_ids, cad, rejected, result):
    _validate_backend(instance)
    requests = cad.batch_size()
    width = 1 + instance.num_speculative_tokens
    context_size = target_token_ids.shape[0]
    if (not isinstance(requests, Integral) or isinstance(requests, bool) or not 1 <= requests <= 8
            or width not in (4, 8, 16) or instance.block_size != 560
            or target_token_ids.ndim != 1 or not 0 < context_size <= 8192
            or not isinstance(cad.max_query_len, Integral) or isinstance(cad.max_query_len, bool)
            or not 0 < cad.max_query_len <= 8192):
        raise RuntimeError('Unsupported DFlash physical-slot request geometry')
    starts, seq, table = cad.query_start_loc, cad.seq_lens, cad.block_table_tensor
    context, query = instance._context_slot_mapping_buffer, instance._slot_mapping_buffer
    tensors = (starts, seq, table, context, query)
    if (any(not isinstance(t, torch.Tensor) or t.device.type != 'cuda' for t in tensors)
            or any(t.device != table.device for t in tensors)
            or starts.shape != (requests+1,) or seq.shape != (requests,)
            or table.ndim != 2 or table.shape[0] < requests or table.shape[1] <= 0
            or starts.dtype != torch.int32 or seq.dtype != torch.int32 or table.dtype != torch.int32
            or context.dtype != torch.int64 or query.dtype != torch.int64
            or context.ndim != 1 or query.ndim != 1
            or context.numel() < context_size or query.numel() < requests*width
            or starts.stride(0) != 1 or seq.stride(0) != 1
            or context.stride(0) != 1 or query.stride(0) != 1
            or table.stride(1) != 1 or table.stride(0) < table.shape[1]):
        raise RuntimeError('Unsupported DFlash physical-slot tensor layout')
    if rejected is not None and (not isinstance(rejected, torch.Tensor)
            or rejected.shape != (requests,) or rejected.dtype != torch.int32
            or rejected.device != table.device or rejected.stride(0) != 1):
        raise RuntimeError('Unsupported DFlash physical-slot rejected-tail metadata')
    if (not isinstance(result, tuple) or len(result) != 3 or result[0] != requests*width
            or result[2].slot_mapping.shape != (requests*width,)
            or result[2].slot_mapping.data_ptr() != query.data_ptr()
            or instance._per_group_slot_mappings[instance.kv_cache_gid].data_ptr() != query.data_ptr()):
        raise RuntimeError('DFlash physical-slot query metadata does not alias the retained buffer')
    return requests, width, context_size


def _wrap_inputs(instance, previous):
    @wraps(previous)
    def inputs(self, target_token_ids, next_token_ids, target_positions,
               target_hidden_states, token_indices_to_sample, cad, num_rejected_tokens_gpu):
        if self is not instance:
            raise RuntimeError('DFlash physical-slot wrapper called on another proposer')
        result = previous(target_token_ids, next_token_ids, target_positions,
                          target_hidden_states, token_indices_to_sample, cad,
                          num_rejected_tokens_gpu)
        requests, width, context_size = _validate_inputs(
            self, target_token_ids, cad, num_rejected_tokens_gpu, result)
        block = 256
        _physical_slots[(requests, triton.cdiv(cad.max_query_len+width, block))](
            cad.query_start_loc, cad.seq_lens, cad.block_table_tensor,
            num_rejected_tokens_gpu if num_rejected_tokens_gpu is not None else 0,
            self._context_slot_mapping_buffer, self._slot_mapping_buffer,
            cad.block_table_tensor.stride(0), cad.block_table_tensor.shape[1],
            self.block_size, width, context_size, num_rejected_tokens_gpu is not None,
            block, num_warps=4)
        return result
    return inputs


def install(worker, evidence_dir=None):
    from vllm.v1.spec_decode import dflash, utils
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
    from ornith_g256 import dflash_spec, dflash_conv_boundary
    runner = worker.model_runner
    draft = runner.drafter
    config = worker.vllm_config
    spec = config.speculative_config
    if (spec is None or spec.method != 'dflash' or spec.num_speculative_tokens != 15
            or not getattr(draft, 'is_dflash2', False) or draft.uses_mrope
            or type(draft) is not dflash.DFlashProposer
            or config.use_v2_model_runner or config.scheduler_config.async_scheduling
            or not 1 <= config.scheduler_config.max_num_seqs <= 8
            or NULL_BLOCK_ID != 0 or PAD_SLOT_ID != -1):
        raise RuntimeError('DFlash physical slots require the retained synchronous IU4 DFlash2 group')
    backend_initialized = _validate_backend(draft, allow_pending=True)
    sources = _source_contract(dflash, utils, dflash_spec, dflash_conv_boundary)
    prior = getattr(draft, _STATE, None)
    if prior is not None:
        if draft.set_inputs_first_pass.__func__ is not prior['wrapper'] or prior['sources'] != sources:
            raise RuntimeError('DFlash physical-slot instance binding changed after installation')
        return prior['receipt']
    previous = draft.set_inputs_first_pass
    if (not isinstance(previous, MethodType) or previous.__self__ is not draft
            or previous.__func__ is not dflash.DFlashProposer.set_inputs_first_pass):
        raise RuntimeError('Another extension replaced this DFlash input instance')
    wrapper = _wrap_inputs(draft, previous)
    receipt = dict(schema='ciru.ornith.iu4.dflash-physical-slots.binding.v1',
        pid=os.getpid(), adapter_sha256=_sha(__file__), runtime_sources=sources,
        query_widths=[4,8,16], requests=[1,8], physical_page_size=560,
        backend_initialized_at_install=backend_initialized,
        backend_validated_before_each_slot_rewrite=True,
        null_block_id=0, padding_slot_id=-1, retired_or_unallocated_pages_masked=True,
        install_before_compile=True, instance_input_hook=True,
        native_input_and_boundary_hooks_preserved=True, per_group_query_mapping_preserved=True,
        target_rope_and_draft_rope_changed=False, target_numerics_changed=False,
        model_weights_changed=False, native_libraries_changed=False,
        allocation_and_logical_windows_changed=False, sampling_changed=False,
        concurrent_k0_and_prefix_guards_changed=False, k0_context_input_hook_preserved=True,
        gpu_numerical_qualified=False, serving_qualified=False)
    draft.set_inputs_first_pass = MethodType(wrapper, draft)
    setattr(draft, _STATE, {'wrapper': wrapper, 'previous': previous,
                          'sources': sources, 'receipt': receipt})
    if evidence_dir is not None:
        path = Path(evidence_dir)/'dflash-physical-slots-binding.json'
        with path.open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
    print('ORNITH_IU4_DFLASH_PHYSICAL_SLOTS '+json.dumps(receipt, sort_keys=True), flush=True)
    return receipt
