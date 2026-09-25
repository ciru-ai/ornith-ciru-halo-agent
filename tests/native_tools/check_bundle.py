#!/usr/bin/env python3
"""Verify composed files and actual XGrammar mapping, without loading a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bundle=a.bundle.resolve();build=json.loads((bundle/'BUILD.json').read_text())
    for name,h in build['files'].items():assert sha(bundle/name)==h,name
    sys.path.insert(0,str(bundle/'plugin-site'))
    import ornith_g256
    import xgrammar as xgr
    grammar=xgr.Grammar.from_ebnf('root ::= [a-z]{129,256}')
    assert grammar is not None
    package=Path(ornith_g256.__file__).parent
    assert package==bundle/'plugin-site/ornith_g256'
    native=package/'_xgrammar_native/libxgrammar-candidate.so'
    selection=json.loads((native.parent/'selection.json').read_text());assert sha(native)==selection['sha256']
    mappings=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libxgrammar' in line))
    assert mappings==[str(native)],mappings
    manifest=json.loads((package/'_vllm_correctness/manifest.json').read_text())
    assert len(manifest['modules'])>=5
    # Resolve the overlay through the same import machinery as an API worker.
    from vllm.entrypoints.openai.chat_completion import serving
    entry=manifest['modules']['vllm.entrypoints.openai.chat_completion.serving']
    assert Path(serving.__file__).resolve()==package/'_vllm_correctness'/entry['file']
    assert sha(serving.__file__)==entry['sha256']
    record=dict(status='PASS_PACKAGED_IMPORT',build_sha256=sha(bundle/'BUILD.json'),
        included=build['included'],native_mappings=mappings,native_sha256=sha(native),
        r03_loaded_file=serving.__file__,r03_sha256=sha(serving.__file__),
        r01_sha256=sha(package/'optimized/grammar_parallel.py'),gpu_model_loaded=False,
        parent_files_verified=len(build['files']),correctness_modules=manifest['modules'])
    a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
