"""Exact mask/acceptance snapshots for independent native-library processes."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import torch
import xgrammar as xgr
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
mode = sys.argv[1]
loaded_libraries = sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libxgrammar_bindings.so' in line))
expected_library = root / f'native-builds/{mode}/xgrammar/libxgrammar_bindings.so'
assert loaded_libraries == [str(expected_library)], ('Wrong native library', loaded_libraries)
loaded_hash = hashlib.sha256(expected_library.read_bytes()).hexdigest()
receipt = next(r for r in json.loads((root/'native-builds/build-receipt.json').read_text()) if r['mode']==mode)
assert loaded_hash == receipt['sha256']
torch.set_num_threads(1)
model = '/srv/llm/work/apodex-ciru-v3-20260911/bundle/models/target'
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
cfg = json.loads(Path(model, 'config.json').read_text())
vocab = cfg.get('text_config', cfg)['vocab_size']
info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab)
compiler = xgr.GrammarCompiler(info, max_threads=4, cache_limit_bytes=512*1024*1024)
grammar = xgr.Grammar.deserialize_json((root / 'review-grammar.json').read_text())
begin = time.perf_counter()
ctx = compiler.compile_grammar(grammar)
compile_seconds = time.perf_counter()-begin
begin = time.perf_counter()
cached = compiler.compile_grammar(grammar)
cache_seconds = time.perf_counter()-begin
cache_bytes = compiler.get_cache_size_bytes()
assert cache_bytes >= ctx.memory_size_bytes, 'The compiled grammar must fit in the configured cache'
assert cache_seconds < max(0.5, compile_seconds * 0.02), 'Repeated compilation must reuse cached work'
print(json.dumps({'mode':mode,'compiled':True,'compile_seconds':compile_seconds,'cache_seconds':cache_seconds,'cache_bytes':cache_bytes,'memory_bytes':ctx.memory_size_bytes}),flush=True)
rows = []
timings = []

def digest(m):
    mask = xgr.allocate_token_bitmask(1, vocab)
    begin = time.perf_counter()
    m.fill_next_token_bitmask(mask)
    timings.append((time.perf_counter()-begin)*1000)
    return hashlib.sha256(mask.numpy().tobytes()).hexdigest()

def inspect(name, context, prefix, suffixes=()):
    m = xgr.GrammarMatcher(context, max_rollback_tokens=32)
    accepted = m.accept_string(prefix)
    row = {'case':name, 'accepted':accepted}
    if accepted:
        row.update(mask=digest(m), completed=m.is_completed(), terminated=m.is_terminated())
        row['suffixes'] = {s: m.fork().accept_string(s) for s in suffixes}
    rows.append(row)

call = '<tool_call>\n<function=review_peer>\n<parameter=layer>geometry</parameter>\n<parameter=verdict>revise</parameter>\n<parameter=assessment>'
suggestion = call + 'The metal surface has sufficient contrast.</parameter>\n<parameter=suggestion>'
for field, prefix, low, high in [('assessment',call,10,1400),('suggestion',suggestion,0,1800)]:
    for n in sorted(set([0,1,low-1,low,low+1,126,127,128,129,high-1,high,high+1])):
        if n < 0: continue
        inspect(f'{field}-ascii-{n}',ctx,prefix+'x'*n,['x','</parameter>','😀'])
    for symbol in ['é','漢','😀']:
        for n in [low,127,128,high-1,high,high+1]:
            inspect(f'{field}-{symbol}-{n}',ctx,prefix+symbol*n,['x',symbol,'</parameter>'])

stream = 'to give the material a satin finish. Keep the dashboard typography clear.\nUnicode: é, 漢字, 😀. Backslash \\ and quoted "text".'
tokens = tok.encode(stream, add_special_tokens=False)
m = xgr.GrammarMatcher(ctx, max_rollback_tokens=32)
assert m.accept_string(suggestion+'Adjust the roughness ')
for i, token in enumerate(tokens):
    before = digest(m)
    probe = tokens[i:i+min(15,len(tokens)-i)]
    for t in probe: assert m.accept_token(t), (i,t)
    m.rollback(len(probe))
    assert digest(m) == before, ('rollback',i)
    assert m.accept_token(token)
    rows.append({'case':f'stream-{i}','before':before,'after':digest(m)})
ending = '</parameter>\n</function>\n</tool_call>'
inspect('complete-native-frame',ctx,suggestion+'A clear suggestion.'+ending,['<|im_end|>'])
inspect('strict-enum',ctx,'<tool_call>\n<function=read_guide>\n<parameter=topic>',['role','bad-topic'])

# Evaluate independent speculative windows concurrently, rollback all drafts,
# and then commit one token. C8 is the workload; C1/C2 check smaller batches.
for c in [1,2,8]:
    for draft_count in [4,7,15]:
        def window(i):
            m = xgr.GrammarMatcher(ctx,max_rollback_tokens=32)
            assert m.accept_string(suggestion+'Adjust the roughness ')
            before = digest(m)
            count = draft_count if i % 2 == 0 else i % (draft_count+1)
            values = []
            for token in tokens[:count]:
                values.append(digest(m)); assert m.accept_token(token)
            if count: m.rollback(count)
            assert digest(m) == before
            assert m.accept_token(tokens[0])
            return {'draft_masks':values,'restored':before,'committed':digest(m)}
        with ThreadPoolExecutor(max_workers=c) as pool: states=list(pool.map(window,range(c)))
        rows.append({'case':f'C{c}-DF{draft_count}-ragged','states':states})

# Explicit min/max, Unicode, escaped JSON, complex/unbounded/large repeats,
# and exhausting the finite-expansion budget all retain the same language.
ebnfs = [
    ('unicode', 'root ::= "[" [^<>]{10,129} "]"', ['['+'é'*9,'['+'😀'*128,'['+'漢'*129,'['+'a'*130]),
    ('exact', 'root ::= [a-z]{130,130}', ['a'*129,'a'*130,'a'*131]),
    ('complex', 'root ::= ("ab" | "cd"){0,129}', ['ab'*128,'cd'*129,'ab'*130]),
    ('unbounded', 'root ::= [a-z]{129,}', ['a'*128,'a'*129,'a'*400]),
    ('large', 'root ::= [a-z]{0,4097}', ['a'*4096,'a'*4097,'a'*4098]),
    ('budget', 'root ::= [a]{0,4096} "|" [b]{0,4096} "|" [c]{0,4096}', ['a'*4096+'|'+'b'*4096+'|'+'c'*4095]),
]
for name, ebnf, prefixes in ebnfs:
    cc=compiler.compile_grammar(ebnf)
    for i,prefix in enumerate(prefixes): inspect(f'{name}-{i}',cc,prefix,['a',']','<|im_end|>'])
schema={'type':'object','properties':{'value':{'type':'string','minLength':3,'maxLength':129}},'required':['value'],'additionalProperties':False}
cc=compiler.compile_json_schema(schema)
for i,value in enumerate(['a'*128,'é'*129,'😀'*130,'quoted "text" and backslash \\ newline\n']):
    inspect(f'json-escaped-{i}',cc,'{"value": '+json.dumps(value,ensure_ascii=False)[:-1],['"}','x'])

result={'mode':mode,'passed':True,'library_package':str(Path(xgr.__file__).parent),
        'native_library':str(expected_library),'native_sha256':loaded_hash,
        'compile_seconds':compile_seconds,'cache_seconds':cache_seconds,
        'cache_bytes':cache_bytes,
        'memory_bytes':ctx.memory_size_bytes,'rows':rows,
        'fill_median_ms':statistics.median(timings),'mask_count':len(timings)}
(root/f'native-ranges-{mode}.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
