#!/usr/bin/env python3
"""Run supplied equivalence tests with explicit receiving tokenizer/native paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-build',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.absolute();out.mkdir(parents=True,exist_ok=False)
    import importlib.metadata
    package=importlib.metadata.distribution('xgrammar')
    original=Path(package.locate_file('xgrammar'))
    assert package.version=='0.2.3'
    builds=json.loads((a.native_build/'build-receipt.json').read_text())
    (out/'native-builds').mkdir();(out/'native-builds/build-receipt.json').write_text(json.dumps(builds,indent=2)+'\n')
    for mode in ['control','candidate']:
        record=next(r for r in builds if r['mode']==mode)
        library=a.native_build/record['library'];assert hashlib.sha256(library.read_bytes()).hexdigest()==record['sha256']
        dest=out/'native-builds'/mode/'xgrammar'
        shutil.copytree(original,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(library,dest/'libxgrammar_bindings.so')
        metadata=Path(package._path);shutil.copytree(metadata,dest.parent/metadata.name)
    source=(HERE/'upstream/test_native_ranges.py').read_text()
    old="model = '/srv/llm/work/apodex-ciru-v3-20260911/bundle/models/target'"
    assert source.count(old)==1;source=source.replace(old,'model = '+repr(str(a.model.resolve())))
    (out/'test_native_ranges.py').write_text(source)
    shutil.copy2(HERE/'fixtures/review-grammar.json',out/'review-grammar.json')
    results={}
    for mode in ['control','candidate']:
        env=dict(os.environ,PYTHONPATH=str(out/'native-builds'/mode),VLLM_PLUGINS='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        with (out/(mode+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(out/'test_native_ranges.py'),mode],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        results[mode]=json.loads((out/f'native-ranges-{mode}.json').read_text())
    assert results['control']['rows']==results['candidate']['rows'],'Native accepted-language/mask difference'
    assert all(r['passed'] and len(r['rows'])==122 and r['mask_count']==461 for r in results.values())
    summary=dict(status='PASS_RECEIVING_NATIVE_EQUIVALENCE',model=str(a.model),cases=122,masks_per_build=461,
        modes={m:{k:v for k,v in r.items() if k!='rows'} for m,r in results.items()},
        test_source_sha256=hashlib.sha256((HERE/'upstream/test_native_ranges.py').read_bytes()).hexdigest(),
        adaptation='Only receiving tokenizer/model path; supplied test body unchanged',no_serving_speed_claim=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
