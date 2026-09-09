"""K0 skips query forwards but preserves DFlash target-context K/V for reentry."""
# Copyright 2026 Ciru.
from functools import wraps

_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch
    from vllm.v1.spec_decode.dflash import DFlashProposer
    original = DFlashProposer.propose

    @wraps(original)
    def propose(self, num_speculative_tokens, *args, **kwargs):
        if num_speculative_tokens != 0:
            return original(self, num_speculative_tokens, *args, **kwargs)
        if args:
            raise ValueError('The isolated K0 context path expects runner keyword arguments')
        # The six-layer query network is not needed to cache target context.
        # Preserve the same combine -> accepted-context slot prep -> fused K/V
        # insert path as upstream. Rejected verification tails get the existing
        # negative slot mapping and are never inserted as accepted context.
        self._last_draft_probs = None
        hidden = self.model.combine_hidden_states(kwargs['target_hidden_states'])
        # Use the already-supported Q8 input-preparation geometry. Query slots
        # and masks are only scratch here: there is no query forward or sample.
        # This avoids inventing a Q1 convolution path just to maintain context.
        self.num_speculative_tokens = 7
        try:
            self.set_inputs_first_pass(
                target_token_ids=kwargs['target_token_ids'],
                next_token_ids=kwargs['next_token_ids'],
                target_positions=kwargs['target_positions'],
                target_hidden_states=hidden,
                token_indices_to_sample=kwargs['token_indices_to_sample'],
                cad=kwargs['common_attn_metadata'],
                num_rejected_tokens_gpu=kwargs.get('num_rejected_tokens_gpu'),
            )
            count = self._dflash_num_context
            self.model.precompute_and_store_context_kv(
                self._dflash_hidden_states,
                self._context_positions_buffer[:count],
                self._context_slot_mapping_buffer[:count],
            )
        finally:
            self.num_speculative_tokens = 0
        count = getattr(self, '_ornith_context_only_calls', 0) + 1
        self._ornith_context_only_calls = count
        if count <= 2:
            print('ORNITH_K0_CONTEXT query forward skipped; target context KV maintained', flush=True)
        ids = kwargs['next_token_ids']
        return torch.empty((ids.numel(), 0), dtype=torch.int64, device=ids.device)

    DFlashProposer.propose = propose
    _INSTALLED = True
