#!/usr/bin/env python3
"""Recheck unchanged R01 with the composed native R04 on real DF7/DF15 windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    package=a.bundle.resolve()/'plugin-site/ornith_g256'
    shutil.copy2(HERE/'upstream/test_ornith_v2.py',out/'test_ornith_v2.py')
    shutil.copy2(HERE/'fixtures/schemas.json',out/'schemas.json')
    shutil.copy2(package/'optimized/grammar_parallel.py',out/'grammar_parallel.py')
    # Load only the pinned native selector before the unmodified compatibility
    # test: importing the plugin's register() would replace its serial oracle.
    driver="""import importlib.util, pathlib, runpy, sys, json
root=pathlib.Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('native_selector',root/'xgrammar_native.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.install()
runpy.run_path('test_ornith_v2.py',run_name='__main__')
mappings=sorted(set(line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libxgrammar' in line))
assert mappings==[str(root/'_xgrammar_native/libxgrammar-candidate.so')],mappings
pathlib.Path('native-mappings.json').write_text(json.dumps(mappings,indent=2)+'\\n')
"""
    (out/'driver.py').write_text(driver)
    with (out/'test.log').open('x') as log:
        subprocess.run([sys.executable,'driver.py',str(package)],cwd=out,
            env=dict(os.environ,VLLM_PLUGINS='',PYTHONPATH=str(out),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    result=json.loads((out/'ornith-compatibility.json').read_text())
    assert result['passed'] and len(result['real_cases'])==12 and result['fake_cases']==8
    result.update(native_mappings=json.loads((out/'native-mappings.json').read_text()),
        source_sha256=hashlib.sha256((out/'test_ornith_v2.py').read_bytes()).hexdigest(),
        build_sha256=hashlib.sha256((a.bundle/'BUILD.json').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
