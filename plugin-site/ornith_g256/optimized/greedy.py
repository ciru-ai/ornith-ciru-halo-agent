"""Install the exact greedy walk beneath the unchanged public prefill wrapper."""
import ast, copy, hashlib, inspect, json, os
from pathlib import Path
import torch
from .greedy_kernel import greedy_path
from ornith_g256.worker import OrnithG256Worker as Parent

EXPECTED_SOURCE='56b817daecb3740cad899a3bc1ebc481def465457d5e1a7a516e0535522e9e7f'

def install(out):
    from ornith_g256 import prefill_draft
    from vllm.v1.spec_decode.dflash import DFlashProposer
    prefill_draft.install()
    assert DFlashProposer._sample_draft_tokens is prefill_draft._sample_draft_tokens
    original=prefill_draft._upstream_sample
    source=Path(inspect.getsourcefile(original))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==EXPECTED_SOURCE
    assert original.__module__=='vllm.v1.spec_decode.dflash'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DFlashProposer')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_sample_draft_tokens'))
    fn.decorator_list=[]
    index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='rows' for t in n.targets))
    fn.body.insert(index,ast.parse('if sampling_metadata.all_greedy:\n return qualified_path(scores, candidate_ids), None').body[0])
    fn.name='_s13_sample'
    out=Path(out)
    globals_=dict(original.__globals__,qualified_path=greedy_path)
    module=ast.fix_missing_locations(ast.Module(body=ast.parse('from __future__ import annotations').body+[fn],type_ignores=[]))
    exec(compile(module,str(source)+':s13', 'exec'),globals_)
    candidate=globals_['_s13_sample']
    def dispatch(self,hidden,metadata):
        if self.is_dflash2 and metadata.all_greedy:
            return candidate(self,hidden,metadata)
        return original(self,hidden,metadata)
    prefill_draft._upstream_sample=dispatch
    with (out/'sampler-installed.json').open('x') as f:
        json.dump(dict(source_sha256=EXPECTED_SOURCE,prefill_wrapper_preserved=True,
                       non_greedy_original=True,delta='Greedy early return after unchanged score calculation'),f,indent=2)

