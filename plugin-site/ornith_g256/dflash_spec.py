"""Keep DFlash query slots after the runner supplies target context metadata."""
# Copyright 2026 Ciru.
from vllm.v1.spec_decode.dflash import DFlashProposer


_upstream_set_inputs_first_pass = DFlashProposer.set_inputs_first_pass


def set_inputs_first_pass(self, *args, **kwargs):
    # The runner seeds per-group metadata with target/context slots. DFlash
    # then generates separate context and query slots from the draft block
    # table. The base proposer otherwise prefers the stale context slots to
    # the new query view, overwriting the query mapping (and overflowing its
    # 64-token buffer on a 128-token prefill).
    if any(group.kv_cache_group_id != self.kv_cache_gid
           for group in self.draft_attn_groups):
        raise ValueError('Ornith DFlash requires one draft KV cache group')
    result = _upstream_set_inputs_first_pass(self, *args, **kwargs)
    _, _, query_metadata = result
    self._per_group_slot_mappings[self.kv_cache_gid] = query_metadata.slot_mapping
    return result


def install():
    """Patch only DFlash's input hook; leave installed vLLM files untouched."""
    current = DFlashProposer.set_inputs_first_pass
    if current not in (_upstream_set_inputs_first_pass, set_inputs_first_pass):
        raise RuntimeError('Another extension replaced DFlash input preparation')
    DFlashProposer.set_inputs_first_pass = set_inputs_first_pass
