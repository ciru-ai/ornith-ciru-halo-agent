#!/usr/bin/env python3
"""Inspect native payload, load guards, entry points and licenses in a wheel."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser();p.add_argument('wheel',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
with zipfile.ZipFile(a.wheel) as z:
    names=z.namelist()
    name=next(n for n in names if n.endswith('/ornith_g256/_xgrammar_native/libxgrammar-candidate.so') or n=='ornith_g256/_xgrammar_native/libxgrammar-candidate.so')
    root=name.split('ornith_g256/',1)[0]+'ornith_g256/'
    selection=json.loads(z.read(root+'_xgrammar_native/selection.json'));receipt=json.loads(z.read(root+'_xgrammar_native/build-receipt.json'))
    digest=hashlib.sha256(z.read(name)).hexdigest();assert digest==selection['sha256']==receipt['sha256']
    manifest=json.loads(z.read(root+'_vllm_correctness/manifest.json'))
    for entry in manifest['modules'].values():assert hashlib.sha256(z.read(root+'_vllm_correctness/'+entry['file'])).hexdigest()==entry['sha256']
    assert any(n.endswith('/licenses/LICENSES/XGRAMMAR-LICENSE') for n in names)
    tag=z.read(next(n for n in names if n.endswith('/WHEEL'))).decode()
    assert 'Root-Is-Purelib: false' in tag and 'linux_x86_64' in tag and '-any' not in tag
    entry=z.read(next(n for n in names if n.endswith('/entry_points.txt'))).decode();assert 'ornith_g256:register' in entry
    assert hashlib.sha256(z.read(root+'grammar_parallel.py')).hexdigest()=='e9c3608438712521980d7fcd60f58862d0b0a2876620183f304c64a03817edea'
record=dict(status='PASS_COMPONENT_WHEEL_CONTENTS',wheel=str(a.wheel),sha256=hashlib.sha256(a.wheel.read_bytes()).hexdigest(),native_sha256=digest,
    correctness_modules=len(manifest['modules']),wheel_metadata=tag,component_only=True,complete_upgraded_model_is_unified_bundle=True)
a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
