"""Candidate v2 semantic guard, including Ornith's dynamic decode widths.

GDN capture buffers are valid only for batches built as pure speculative
decode. Equal token counts alone do not establish that contract. This guard
records the same scheduled draft counts used by the installed metadata builder
and rejects prompt/mixed/unverified batches before uniform graph dispatch.

All speculative methods and widths use the same contract; ordinary decode,
Q1/non-speculation, and explicit dummy/capture dispatch keep their prior path.
No model arithmetic, cache ownership, sampler, or installed vLLM files change.

Install before ornith_g256.dynamic_graphs.install. The dynamic wrapper must be
outside this guard so its temporary effective width is visible at dispatch.
Input preparation can have a different configured maximum width; the actual
scheduled counts and draft counts, not that maximum, establish compatibility.
"""
from functools import wraps
import inspect
import json
import sys

_installed = False
_ATTR = '_ciru_hybrid_graph_phase_receipt'

def make_wrappers(original_prepare, original_determine):
    expected_prepare = ('self', 'scheduler_output', 'num_scheduled_tokens')
    expected_determine = ('self', 'num_tokens', 'num_reqs', 'num_scheduled_tokens_np',
        'max_num_scheduled_tokens', 'use_cascade_attn', 'allow_microbatching',
        'force_eager', 'force_uniform_decode', 'force_has_lora',
        'force_num_active_loras', 'num_encoder_reqs')
    if tuple(inspect.signature(original_prepare).parameters) != expected_prepare:
        raise RuntimeError('Unsupported hybrid graph input preparation signature')
    if tuple(inspect.signature(original_determine).parameters) != expected_determine:
        raise RuntimeError('Unsupported hybrid graph dispatch signature')

    def scoped(self):
        return bool(self.model_config.is_hybrid and self.uniform_decode_query_len > 1)

    @wraps(original_prepare)
    def prepare(self, scheduler_output, num_scheduled_tokens):
        setattr(self, _ATTR, None)
        result = original_prepare(self, scheduler_output, num_scheduled_tokens)
        if scoped(self):
            batch = self.input_batch
            n = batch.num_reqs
            reqs = tuple(batch.req_ids)
            counts = tuple(int(x) for x in num_scheduled_tokens)
            computed = tuple(int(x) for x in batch.num_computed_tokens_cpu[:n])
            prompts = tuple(int(x) for x in batch.num_prompt_tokens[:n])
            drafts = scheduler_output.scheduled_spec_decode_tokens
            draft_counts = tuple(len(drafts[r]) if r in drafts else -1 for r in reqs)
            setattr(self, _ATTR, dict(reqs=reqs, counts=counts, computed=computed,
                prompts=prompts, draft_counts=draft_counts,
                width=int(self.uniform_decode_query_len)))
        return result

    @wraps(original_determine)
    def determine(self, num_tokens, num_reqs, num_scheduled_tokens_np,
                  max_num_scheduled_tokens, use_cascade_attn,
                  allow_microbatching=True, force_eager=False,
                  force_uniform_decode=None, force_has_lora=None,
                  force_num_active_loras=None, num_encoder_reqs=0):
        reason = None
        evidence = None
        # _dummy_run explicitly supplies this flag. Its synthetic request
        # arrays and metadata are intentionally not interpreted as real work.
        width = self.uniform_decode_query_len
        if (force_uniform_decode is None and scoped(self) and num_reqs > 0
                and max_num_scheduled_tokens == width
                and num_tokens == width * num_reqs):
            b = self.input_batch
            receipt = getattr(self, _ATTR, None)
            counts = tuple(int(x) for x in num_scheduled_tokens_np)
            computed = tuple(int(x) for x in b.num_computed_tokens_cpu[:num_reqs])
            prompts = tuple(int(x) for x in b.num_prompt_tokens[:num_reqs])
            if (receipt is None or b.num_reqs != num_reqs
                    or len(counts) != num_reqs or any(x != width for x in counts)
                    or len(computed) != num_reqs or len(prompts) != num_reqs
                    or receipt['reqs'] != tuple(b.req_ids)
                    or receipt['counts'] != counts
                    or receipt['computed'] != computed or receipt['prompts'] != prompts):
                reason = 'unverified-current-batch'
            elif any(c < p for c, p in zip(computed, prompts)):
                reason = 'prompt-or-mixed-phase'
            elif any(k != width - 1 for k in receipt['draft_counts']):
                reason = 'missing-or-incompatible-draft-metadata'
            if reason is not None:
                force_uniform_decode = False
                evidence = dict(reason=reason, width=int(width),
                    num_tokens=int(num_tokens), num_reqs=int(num_reqs),
                    computed=computed, prompt_tokens=prompts, scheduled=counts,
                    draft_counts=None if receipt is None else receipt['draft_counts'])
        result = original_determine(self, num_tokens, num_reqs,
            num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn,
            allow_microbatching=allow_microbatching, force_eager=force_eager,
            force_uniform_decode=force_uniform_decode, force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs)
        if reason is not None:
            mode = getattr(result[0], 'name', str(result[0]))
            if mode == 'FULL':
                raise RuntimeError('Hybrid graph phase guard could not exclude incompatible FULL replay')
            count = getattr(self, '_ciru_hybrid_graph_phase_guard_count', 0) + 1
            self._ciru_hybrid_graph_phase_guard_count = count
            if count <= 8 or count % 1000 == 0:
                print('CIRU_HYBRID_GRAPH_PHASE_GUARD ' + json.dumps(
                    dict(evidence, graph_mode=mode, count=count)), flush=True)
        return result
    return prepare, determine

def install():
    global _installed
    if _installed:
        return
    dynamic = sys.modules.get('ornith_g256.dynamic_graphs')
    if dynamic is not None and getattr(dynamic, '_INSTALLED', False):
        raise RuntimeError('Install the hybrid graph phase guard before dynamic_graphs.install')
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    cls = GPUModelRunner
    cls._prepare_inputs, cls._determine_batch_execution_and_padding = make_wrappers(
        cls._prepare_inputs, cls._determine_batch_execution_and_padding)
    _installed = True
