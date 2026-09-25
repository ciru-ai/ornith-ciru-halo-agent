"""Exercise the original vLLM control flow and real XGrammar mask equivalence."""
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import numpy as np
import torch
import xgrammar as xgr
from transformers import AutoTokenizer
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
from grammar_parallel import MaskState, make_wrapper

ROOT = Path(__file__).resolve().parent
original = StructuredOutputManager.grammar_bitmask
torch.set_num_threads(1)


class Reasoner:
    def is_reasoning_end(self, tokens): return 99 in tokens
    def is_reasoning_end_streaming(self, tokens, new): return 99 in new


class FakeGrammar:
    def __init__(self, terminated=False):
        self.history = [9] if terminated else [2]
        self.finished = False
        self.fail = False
    def is_terminated(self): return self.history[-1:] == [9]
    def fill_bitmask(self, mask, index):
        if self.fail:
            self.finished = True
            raise ValueError('fixture failure')
        mask[index].fill_(sum(self.history) * 100 + len(self.history))
        self.finished = True
    def accept_tokens(self, req_id, tokens):
        for token in tokens:
            if self.is_terminated(): break
            if token == 7: return False
            self.history.append(token)
        return True
    def validate_tokens(self, tokens):
        result = []
        for token in tokens:
            if token == 7: break
            result.append(token)
            if token == 9: break
        return result
    def rollback(self, count): del self.history[-count:]


def manager(vocab=64, spec=4, diffusion=False):
    m = StructuredOutputManager.__new__(StructuredOutputManager)
    m.vllm_config = NS(num_speculative_tokens=spec, model_config=NS(is_diffusion=diffusion), scheduler_config=NS(max_num_seqs=8))
    m.backend = NS(allocate_token_bitmask=lambda count: xgr.allocate_token_bitmask(count, vocab))
    m._grammar_bitmask = m.backend.allocate_token_bitmask(8 * (spec + 1))
    m._full_mask = torch.tensor(-1, dtype=torch.int32)
    m.fill_bitmask_parallel_threshold = 128
    m.enable_in_reasoning = False
    m.reasoner_cls = Reasoner
    m.tokenizer = None
    m._ciru_mask_state = MaskState(cost_per_request_ns=1e9)
    return m


def requests(factory, count=8, thinking=None):
    return {str(i): NS(structured_output_request=NS(grammar=factory(i), reasoning_ended=not (thinking and i in thinking), reasoner=Reasoner()), prompt_token_ids=[10], all_token_ids=[10, 11]) for i in range(count)}


def shutdown(m):
    if m._ciru_mask_state.pool: m._ciru_mask_state.pool.shutdown()


def check_fake(name, *, count=8, spec=4, drafts=None, thinking=None, terminated=False, diffusion=False, alias=False, allocate=False, expect_parallel=True):
    a, c = manager(spec=spec, diffusion=diffusion), manager(spec=spec, diffusion=diffusion)
    ar = requests(lambda i: FakeGrammar(terminated), count, thinking)
    cr = requests(lambda i: FakeGrammar(terminated), count, thinking)
    if alias:
        ar['1'].structured_output_request.grammar = ar['0'].structured_output_request.grammar
        cr['1'].structured_output_request.grammar = cr['0'].structured_output_request.grammar
    if allocate: a._grammar_bitmask = c._grammar_bitmask = None
    ids = list(ar)
    drafts = drafts or {i: [1, 2, 3, 4][:spec] for i in ids}
    before_a = [list(ar[i].structured_output_request.grammar.history) for i in ids]
    before_c = [list(cr[i].structured_output_request.grammar.history) for i in ids]
    expected = original(a, ar, ids, drafts).copy()
    actual = make_wrapper(original, FakeGrammar)(c, requests=cr, structured_output_request_ids=ids, scheduled_spec_decode_tokens=drafts).copy()
    assert np.array_equal(actual, expected), name
    assert [ar[i].structured_output_request.grammar.history for i in ids] == before_a, name
    assert [cr[i].structured_output_request.grammar.history for i in ids] == before_c, name
    assert bool(c._ciru_mask_state.parallel_batches) == expect_parallel, (name, c._ciru_mask_state)
    shutdown(c)
    print(json.dumps({'case': name, 'mask_equal': True, 'state_restored': True, 'parallel': expect_parallel}), flush=True)


check_fake('C8 MTP4')
check_fake('C1 original path', count=1, expect_parallel=False)
check_fake('C8 non-spec', spec=0)
check_fake('ragged speculative windows', drafts={str(i): [1, 2, 3, 4][:i % 5] for i in range(8)})
check_fake('invalid draft sentinels', drafts={str(i): [1, -1, 2, -1] for i in range(8)})
check_fake('terminated matchers', terminated=True)
check_fake('reasoning ends inside draft window', thinking=set(range(1, 8)), drafts={str(i): ([1, 2, 3, 4] if i == 0 else [5, 99, 7, 3]) for i in range(8)})
check_fake('all reasoning remains serial', thinking=set(range(8)), drafts={str(i): [5, 6, 3, 4] for i in range(8)}, expect_parallel=False)
check_fake('diffusion original path', diffusion=True, expect_parallel=False)
check_fake('aliased matchers remain serial', alias=True, expect_parallel=False)
check_fake('first allocation stays original', allocate=True, expect_parallel=False)
m = manager()
r = requests(lambda i: FakeGrammar())
r['0'].structured_output_request.grammar.fail = True
try:
    make_wrapper(original, FakeGrammar)(m, r, list(r), {})
    raise AssertionError('Expected fixture error')
except ValueError as error:
    assert str(error) == 'fixture failure'
    assert all(q.structured_output_request.grammar.finished for q in r.values())
shutdown(m)
print(json.dumps({'case': 'all writers drained before exception', 'passed': True}), flush=True)

MODEL = '/srv/llm/work/apodex-ciru-v3-20260911/bundle/models/target'
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
cfg = json.loads(Path(MODEL, 'config.json').read_text())
vocab = cfg.get('text_config', cfg)['vocab_size']
compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab), max_threads=8)
tools = json.loads((ROOT / 'schemas.json').read_text())[0]['tools']
for tool in tools: tool['function']['strict'] = True
ctx = compiler.compile_structural_tag(xgr.get_model_structural_tag(model='qwen_3_coder', tools=tools, tool_choice='auto', reasoning=False))
prefix = '<tool_call>\n<function=review_peer>\n<parameter=layer>geometry</parameter>\n<parameter=verdict>revise</parameter>\n<parameter=assessment>The metal surface has sufficient contrast.</parameter>\n<parameter=suggestion>Adjust the roughness '
draft = tokenizer.encode('to give the material a satin finish', add_special_tokens=False)[:4]
def real_grammar(i):
    matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=8)
    assert matcher.accept_string(prefix)
    return XgrammarGrammar(vocab_size=vocab, matcher=matcher, ctx=ctx)
wrapped = make_wrapper(original)
timings = []
for count, window in [(1, 'single'), (8, 'ragged'), (8, 'uniform')]:
    for mode in ['serial', 'parallel', 'parallel', 'serial']:
        a, c = manager(vocab=vocab), manager(vocab=vocab)
        ar, cr = requests(real_grammar, count), requests(real_grammar, count)
        ids = list(ar)
        scheduled = {i: (draft if window == 'uniform' else draft[:int(i) % 5]) for i in ids}
        expected = original(a, ar, ids, scheduled).copy()
        got = wrapped(c, cr, ids, scheduled).copy()
        assert np.array_equal(got, expected)
        for i in ids:
            assert ar[i].structured_output_request.grammar.num_processed_tokens == cr[i].structured_output_request.grammar.num_processed_tokens == 0
        fn = (lambda: original(a, ar, ids, scheduled)) if mode == 'serial' else (lambda: wrapped(c, cr, ids, scheduled))
        fn()
        samples = []
        for _ in range(2):
            begin = time.perf_counter_ns()
            output = fn()
            samples.append((time.perf_counter_ns() - begin) / 1e6)
            assert np.array_equal(output, expected)
        row = {'count': count, 'window': window, 'mode': mode, 'ms': samples, 'median_ms': statistics.median(samples), 'mask_equal': True, 'state_restored': True}
        timings.append(row)
        print(json.dumps(row), flush=True)
        shutdown(c)
enum_prefix = '<tool_call>\n<function=read_guide>\n<parameter=topic>'
complete_call = enum_prefix + 'role</parameter>\n</function>\n</tool_call>'
for case, text, extra in [('strict enum', enum_prefix, []), ('real stop and rollback', complete_call, [tokenizer.eos_token_id])]:
    def factory(i):
        m = xgr.GrammarMatcher(ctx, max_rollback_tokens=8)
        assert m.accept_string(text), (case, text)
        return XgrammarGrammar(vocab_size=vocab, matcher=m, ctx=ctx)
    a, c = manager(vocab=vocab), manager(vocab=vocab)
    ar, cr = requests(factory), requests(factory)
    ids = list(ar)
    schedule = {i: extra for i in ids}
    expected = original(a, ar, ids, schedule).copy()
    actual = wrapped(c, cr, ids, schedule).copy()
    assert np.array_equal(expected, actual), case
    assert all(q.structured_output_request.grammar.num_processed_tokens == 0 for q in cr.values())
    if case == 'strict enum':
        bad = tokenizer.encode('z', add_special_tokens=False)
        assert len(bad) == 1
        assert not (int(actual[0, bad[0] // 32]) & (1 << (bad[0] % 32)))
    shutdown(c)
    print(json.dumps({'case': case, 'mask_equal': True, 'state_restored': True}), flush=True)
(ROOT / 'parallel-tests.json').write_text(json.dumps({'passed': True, 'timings': timings}, indent=2))
