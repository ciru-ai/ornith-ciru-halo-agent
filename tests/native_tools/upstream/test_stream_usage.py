"""Run the actual original/patched async serving method against buffered output.

Transport/model collaborators are small fixtures; the complete token accounting,
parser gate, metadata suppression, final/error handling and yield logic are the
unmodified installed method except for the candidate's one conditional change.
"""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
import __future__
from stream_usage import patch_source

ROOT = Path(__file__).resolve().parent

class Record(NS):
    def __getattr__(self, key): return None
    def model_dump_json(self, **kwargs):
        return json.dumps(self, default=lambda x: vars(x))

class NamedChoice: pass
class GenerationError(Exception): pass

class BufferedParser:
    invalid = False
    def __init__(self, *args, **kwargs): pass
    def count_reasoning_tokens(self, tokens): return sum(t < 14 for t in tokens)
    def parse_delta(self, *, finished, **kwargs):
        if not finished: return None
        if self.invalid: raise ValueError('Incomplete native tool frame')
        return Record(tool_calls=[Record(index=0, id='fixture-call', type='function', function=Record(name='capture', arguments='{"data":[1,2,3]}'))])

def compile_method(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OpenAIServingChat')
    fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'chat_completion_stream_generator')
    ns = dict(
        time=NS(time=lambda: 0), ChatCompletionNamedToolChoiceParam=NamedChoice,
        ChatCompletionResponseStreamChoice=Record, ChatCompletionStreamResponse=Record,
        DeltaMessage=Record, UsageInfo=Record, PerRequestMetrics=Record,
        should_include_usage=lambda opts, force: (opts.include_usage or force, (opts.include_usage or force) and opts.continuous_usage_stats),
        as_list=list, maybe_filter_parallel_tool_calls=lambda choice, request: choice,
        _make_completion_tokens_details=lambda count: {'reasoning_tokens': count},
        _make_prompt_tokens_details=lambda *args: {'cached_tokens': 0},
        build_spec_decoding_metrics=lambda res: None,
        GenerationError=GenerationError, logger=NS(exception=lambda *args: None),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<installed-serving-method>', 'exec', flags=__future__.annotations.compiler_flag), ns)
    return ns[fn.name]

original = (ROOT / 'baseline/chat_completion_serving.py').read_text()
baseline, candidate = map(compile_method, [original, patch_source(original)])

async def run(method, *, continuous, usage=True, hidden=False, token_ids=False, n=1, invalid=False):
    parser = type('CaseParser', (BufferedParser,), {'invalid': invalid})
    self = NS(
        parser_cls=parser, model_config=None, enable_force_include_usage=False,
        _include_reasoning_tokens_details=True, enable_log_outputs=False,
        request_logger=None, enable_prompt_tokens_details=True, enable_per_request_metrics=False,
        system_fingerprint='fixture', get_chat_request_role=lambda request: 'assistant',
        _create_chat_logprobs=lambda **kw: {'content': [{'token': 'PRIVATE_BUFFER_TOKEN', 'logprob': -1}]},
        _raise_if_error=lambda *args: None,
        create_streaming_error_response=lambda error: json.dumps({'error': str(error)}),
    )
    request = Record(n=n, tool_choice='auto', tools=[], stream_options=Record(include_usage=usage, continuous_usage_stats=continuous), return_prompt_text=False, return_token_ids=token_ids, echo=False, logprobs=True, top_logprobs=1, logprob_token_ids=None, return_tokens_as_token_ids=False, include_reasoning=not hidden)
    async def outputs():
        for tokens, finished in [([], False), ([11], False), ([12,13,14], False), ([], False), ([15], False), ([16], True)]:
            yield Record(prompt_token_ids=[1,2], encoder_prompt_token_ids=None, num_cached_tokens=0, num_cache_creation_tokens=0, prompt='fixture', metrics=None, outputs=[Record(index=i, text='buffered' if tokens else '', token_ids=tokens, logprobs=[], finish_reason='stop' if finished else None, stop_reason=None) for i in range(n)])
    metadata=Record()
    chunks=[]
    async for event in method(self, request, outputs(), 'fixture', 'fixture-model', [], object(), metadata):
        value=event.removeprefix('data: ').strip()
        chunks.append(value if value == '[DONE]' else json.loads(value))
    return chunks, metadata

def payload(chunks):
    result=[]
    for chunk in chunks:
        if not isinstance(chunk, dict): continue
        for choice in chunk.get('choices', []):
            delta=choice.get('delta', {})
            if delta or choice.get('finish_reason'):
                result.append((choice['index'], delta, choice.get('finish_reason')))
    return result

async def main():
    rows=[]
    for usage, continuous in [(False, False), (True, False), (True, True)]:
        for hidden in [False, True]:
            for token_ids in [False, True]:
                for n in [1, 2]:
                    args=dict(usage=usage, continuous=continuous, hidden=hidden, token_ids=token_ids, n=n)
                    before, bm=await run(baseline, **args)
                    after, cm=await run(candidate, **args)
                    assert payload(before) == payload(after), args
                    assert bm.final_usage_info.completion_tokens == cm.final_usage_info.completion_tokens == 6*n
                    if not continuous: assert before == after, args
                    if continuous:
                        for i in range(n):
                            counts=[c['usage']['completion_tokens'] for c in after if isinstance(c, dict) and c.get('usage') and c.get('choices') and c['choices'][0]['index']==i]
                            assert counts == [0,1,4,5,6] or (token_ids and not hidden and counts == [0,1,4,4,5,6]), (args, counts)
                    if hidden:
                        for c in after:
                            if not isinstance(c,dict): continue
                            for choice in c.get('choices',[]):
                                assert choice.get('token_ids') is None and choice.get('logprobs') is None
                    assert after[-1] == '[DONE]'
                    rows.append({**args, 'passed': True})
    before, _=await run(baseline, continuous=True, invalid=True)
    after, _=await run(candidate, continuous=True, invalid=True)
    assert [c for c in before if isinstance(c,dict) and 'error' in c] == [c for c in after if isinstance(c,dict) and 'error' in c]
    assert not any(c.get('usage') and not c.get('choices') for c in after if isinstance(c,dict))
    assert after[-1]=='[DONE]'
    result={'passed': True, 'cases': rows, 'invalid_frame_still_errors': True, 'note': 'Full original async serving method with deterministic model/parser/serialization collaborators; real native API confirmation remains required.'}
    (ROOT/'stream-usage-tests.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed':True, 'cases':len(rows), 'invalid_frame_still_errors':True}))

asyncio.run(main())
