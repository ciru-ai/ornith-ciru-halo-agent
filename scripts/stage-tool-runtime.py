#!/usr/bin/env python3
"""Add qualified tool changes to an existing plugin without removing its optimizations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
R01 = 'e9c3608438712521980d7fcd60f58862d0b0a2876620183f304c64a03817edea'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-plugin',type=Path,required=True)
    p.add_argument('--native-build',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='Fresh plugin-site directory')
    a=p.parse_args();parent=a.parent_plugin.resolve();out=a.out.absolute()
    if parent.name!='ornith_g256' or not (parent/'__init__.py').is_file():p.error('Expected existing ornith_g256 package')
    if out.exists():p.error('Choose a fresh output directory')
    records=json.loads((a.native_build/'build-receipt.json').read_text())
    record=next(r for r in records if r['mode']=='candidate')
    source_identity=json.loads((ROOT/'patches/xgrammar/SOURCE.json').read_text())
    assert record['source_commit']==source_identity['commit']
    assert record['sources']==source_identity['candidate_source_sha256']
    assert record['patch_sha256']==source_identity['patch_sha256']
    library=a.native_build/record['library'];assert sha(library)==record['sha256']
    out.mkdir(parents=True);dest=out/'ornith_g256'
    shutil.copytree(parent,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    before={str(f.relative_to(parent)):sha(f) for f in parent.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
    manifest_path=dest/'_vllm_correctness/manifest.json'
    manifest=json.loads(manifest_path.read_text());prior=dict(manifest['modules'])
    shipped=json.loads((ROOT/'plugin-site/ornith_g256/_vllm_correctness/manifest.json').read_text())
    name='vllm.entrypoints.openai.chat_completion.serving';entry=shipped['modules'][name]
    if name in prior:assert prior[name]==entry,'Unexpected existing chat overlay'
    source=ROOT/'plugin-site/ornith_g256/_vllm_correctness'/entry['file']
    assert sha(source)==entry['sha256'];shutil.copy2(source,dest/'_vllm_correctness'/entry['file'])
    manifest['modules'][name]=entry;manifest['runtime_version']='1.0.3'
    assert all(manifest['modules'][k]==v for k,v in prior.items())
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    init=dest/'__init__.py';text=init.read_text()
    existing_r01=list(dest.rglob('grammar_parallel.py'))
    assert all(sha(f)==R01 for f in existing_r01),'Review changed R01 before composing runtimes'
    if not existing_r01:
        r01=ROOT/'plugin-site/ornith_g256/grammar_parallel.py';assert sha(r01)==R01
        shutil.copy2(r01,dest/'grammar_parallel.py')
        text+='\n    from .grammar_parallel import install as install_parallel_grammar\n    install_parallel_grammar()\n'
    else:
        assert 'install_parallel_grammar()' in text,'Existing R01 must already be registered'
    loader=ROOT/'plugin-site/ornith_g256/xgrammar_native.py'
    if (dest/'xgrammar_native.py').exists():assert sha(dest/'xgrammar_native.py')==sha(loader),'Review unexpected existing loader'
    shutil.copy2(loader,dest/'xgrammar_native.py')
    if '_install_native_grammar()' not in text:
        anchor='from .runtime_correctness import install as _install_correctness'
        assert text.count(anchor)==1
        text=text.replace(anchor,'from .xgrammar_native import install as _install_native_grammar\n_install_native_grammar()\n\n'+anchor)
    init.write_text(text)
    native=dest/'_xgrammar_native';native.mkdir(exist_ok=True)
    shutil.copy2(library,native/'libxgrammar-candidate.so')
    (native/'selection.json').write_text(json.dumps(dict(mode='candidate',sha256=record['sha256']),indent=2)+'\n')
    (native/'build-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    changed=[n for n,h in before.items() if sha(dest/n)!=h]
    assert set(changed)<={'__init__.py','_vllm_correctness/manifest.json'},changed
    for f in existing_r01:assert sha(f)==R01
    files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
    (out/'TOOL-INTEGRATION.json').write_text(json.dumps(dict(status='STAGED_PENDING_QUALIFICATION',
        parent=str(parent),parent_files=before,changed_existing_files=changed,files=files,
        r01_sha256=R01,r01_existing_preserved=bool(existing_r01),r04=record,r03=entry,
        model_settings_changed=False,weights_changed=False,strict_policy_changed=False),indent=2)+'\n')
    print(json.dumps(dict(status='STAGED',root=str(out),changed_existing_files=changed,r01_preserved=bool(existing_r01))))


if __name__=='__main__':main()
