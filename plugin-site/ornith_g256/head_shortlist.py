"""Exact top-k decode logits from a 4-bit shortlist rescored by the INT8 head.

A G256 4-bit copy of the rotated INT8 head ranks the vocabulary; the best 256
ids per row are rescored with the unchanged INT8 kernel and every other logit is
-inf. The shortlist is used only when each request samples greedily or from at
most 20 tokens without logprobs, bias, allowed ids, bad words, penalties or
structured output, so the sampled distribution equals the full-head one.
"""
import ctypes as C
import torch

VOCAB = 248320
CANDIDATES = 256
DRAFT_VOCAB = 65536
_INSTALLED = False
_STATE = {}


def _bank(layer, device):
    codes, scales = layer.head_tilebank_codes, layer.head_scales
    tiles = codes.shape[0]
    rows = tiles * 16
    bank = torch.empty((rows // 32, 8, 32, 32), dtype=torch.int32, device=device)
    meta = torch.empty((rows // 32, 8, 32), dtype=torch.int32, device=device)
    shifts = torch.arange(8, device=device, dtype=torch.int64) * 4
    for t0 in range(0, tiles, 1024):
        t1 = min(tiles, t0 + 1024)
        r0, r1 = t0 * 16, t1 * 16
        raw = codes[t0:t1].contiguous().view(torch.int8).reshape(t1 - t0, 128, 4, 16, 4)
        w = raw.permute(0, 3, 1, 2, 4).reshape(r1 - r0, 2048).float()
        g = (w * scales[r0:r1].float().repeat_interleave(128, dim=1)).view(r1 - r0, 8, 256)
        low = g.amin(dim=2)
        scale = ((g.amax(dim=2) - low) / 15.0).half()
        offset = low.half()
        q = torch.round((g - offset.float()[..., None]) / scale.float().clamp_min(1e-30)[..., None]).clamp_(0, 15)
        words = (q.to(torch.int64).view(r1 - r0, 8, 32, 8) << shifts).sum(dim=3) & 0xFFFFFFFF
        words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
        bank[r0 // 32:r1 // 32] = words.view((r1 - r0) // 32, 32, 8, 32).permute(0, 2, 3, 1)
        packed = (offset.view(torch.int16).to(torch.int32) << 16) | (scale.view(torch.int16).to(torch.int32) & 0xFFFF)
        meta[r0 // 32:r1 // 32] = packed.view((r1 - r0) // 32, 32, 8).permute(0, 2, 1)
    return bank, meta


def _eligible(runner):
    for req_id in runner.input_batch.req_ids:
        state = runner.requests.get(req_id)
        sp = None if state is None else state.sampling_params
        if sp is None:
            return False
        if sp.temperature >= 1e-5 and not 0 < sp.top_k <= 20:
            return False
        if (sp.logprobs is not None or sp.prompt_logprobs is not None or sp.logit_bias
                or sp.allowed_token_ids or sp.bad_words or getattr(sp, 'structured_outputs', None) is not None
                or getattr(sp, 'logits_processors', None) or sp.frequency_penalty != 0.0
                or sp.presence_penalty != 0.0 or sp.repetition_penalty != 1.0):
            return False
    return True


def _head(method, x, codes, scales, out, vocab):
    from . import native
    runtime = method.runtime
    native.check(native._libs['head'][3](
        native.ptr(x), native.ptr(codes), native.ptr(scales), native.ptr(runtime.workspace),
        runtime.workspace.numel(), native.ptr(out), None, 0, native.ptr(runtime.slots[method.slot]),
        x.shape[0], 64, vocab, runtime.head_geometry, native.stream(x)), 'W8 head shortlist')


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from . import native, dense_n32
    from .method import W8HeadMethod
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
    original_apply = W8HeadMethod.apply
    original_execute = GPUModelRunner.execute_model
    original_candidates = DFlash2Qwen3ForCausalLM.compute_candidates

    def available():
        if 'available' not in _STATE:
            head_lib = native._libs.get('head')
            _STATE['available'] = (head_lib is not None and hasattr(head_lib[1], 'ornith_head_i8_tile_shortlist')
                                   and dense_n32._LIB is not None and hasattr(dense_n32._LIB, 'ornith_dense_g256_n32_rows'))
        return _STATE['available']

    def approximate(layer, x):
        if 'bank' not in _STATE:
            _STATE['bank'] = _bank(layer, x.device)
            layout = (C.c_size_t * 7)()
            native.check(dense_n32._LIB.ornith_dense_g256_get_layout(16, VOCAB, 2048, 8, layout), 'head shortlist layout')
            _STATE['workspace'] = torch.empty(int(layout[0]), dtype=torch.uint8, device=x.device)
            _STATE['flags'] = torch.zeros(1, dtype=torch.int32, device=x.device)
        out = torch.empty((x.shape[0], VOCAB), dtype=x.dtype, device=x.device)
        bank, meta = _STATE['bank']
        workspace = _STATE['workspace']
        native.check(dense_n32._LAUNCH(
            native.ptr(x), native.ptr(bank), native.ptr(meta), native.ptr(workspace), workspace.numel(),
            native.ptr(out), native.ptr(_STATE['flags']), x.shape[0], 16, VOCAB, 2048, 8, 128, 2,
            native.stream(x)), 'head shortlist ranking')
        return out

    def shortlist(method, layer, x):
        rows = x.shape[0]
        ids = approximate(layer, x).topk(CANDIDATES, dim=-1).indices.reshape(-1)
        size = next(s for s in (256, 1024, 4096) if ids.numel() <= s)
        if ids.numel() < size:
            ids = torch.cat([ids, ids[:1].expand(size - ids.numel())])
        tiles = layer.head_tilebank_codes.view(torch.int32).view(-1, 128, 4, 16)
        codes = tiles[ids // 16, :, :, ids % 16].view(size // 16, 16, 128, 4).permute(0, 2, 3, 1).contiguous()
        exact = torch.empty((rows, size), dtype=x.dtype, device=x.device)
        _head(method, x, codes, layer.head_scales[ids].contiguous(), exact, size)
        logits = torch.full((rows, VOCAB), float('-inf'), dtype=x.dtype, device=x.device)
        return logits.scatter_(1, ids.unsqueeze(0).expand(rows, size), exact)

    def apply(self, layer, x, bias=None):
        rows = x.reshape(-1, 2048)
        runner = _STATE.get('runner')
        if (bias is None and self.runtime is not None and 1 <= rows.shape[0] <= 16 and runner is not None
                and not torch.cuda.is_current_stream_capturing() and available() and _eligible(runner)):
            return shortlist(self, layer, rows.contiguous())
        return original_apply(self, layer, x, bias)

    def execute_model(runner, *args, **kwargs):
        _STATE['runner'] = runner
        return original_execute(runner, *args, **kwargs)

    def compute_candidates(self, hidden_states):
        head = self.lm_head
        method = getattr(head, 'quant_method', None)
        if not hasattr(head, 'head_tilebank_codes') or getattr(method, 'runtime', None) is None or not available():
            return original_candidates(self, hidden_states)
        processor = self.candidate_logits_processor
        x = hidden_states.reshape(-1, 2048).contiguous()
        logits = torch.empty((x.shape[0], DRAFT_VOCAB), dtype=x.dtype, device=x.device)
        for start in range(0, x.shape[0], 64):
            _head(method, x[start:start + 64], head.head_tilebank_codes, head.head_scales,
                  logits[start:start + 64], DRAFT_VOCAB)
        values, ids = torch.topk(logits, self.model.candidate_selector.top_k, dim=-1)
        values = values.float()
        if processor.scale != 1.0:
            values = values * processor.scale
        if processor.soft_cap is not None:
            values = torch.tanh(values / processor.soft_cap) * processor.soft_cap
        return ids.to(torch.int64), values

    W8HeadMethod.apply = apply
    GPUModelRunner.execute_model = execute_model
    DFlash2Qwen3ForCausalLM.compute_candidates = compute_candidates
    _INSTALLED = True
