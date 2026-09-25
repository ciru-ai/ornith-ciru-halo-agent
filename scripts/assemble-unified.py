#!/usr/bin/env python3
"""Create one upgraded bundle from the complete promoted parent plus R04/R03."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PROMOTED_PARENT='d72dba3a3d928be3d1207772707774ec42849b7331fab667603f2d2f2233eae9'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def preserve_plugin_metadata(parent_site, output_site):
    """Keep entry-point/distribution metadata and other parent site files."""
    for item in parent_site.iterdir():
        if item.name in {'ornith_g256', '__pycache__'} or item.suffix == '.pyc':
            continue
        dest = output_site / item.name
        if dest.exists() or dest.is_symlink():
            raise RuntimeError(f'Unexpected integration output conflicts with parent: {item.name}')
        if item.is_symlink():
            dest.symlink_to(item.readlink(), target_is_directory=item.is_dir())
        elif item.is_dir():
            shutil.copytree(item, dest, symlinks=True)
        else:
            shutil.copy2(item, dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--native-build',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();parent=a.parent.resolve();out=a.out.absolute()
    assert sha(parent/'BUILD.json')==PROMOTED_PARENT,'Requires complete qualified combined-v3 parent'
    build=json.loads((parent/'BUILD.json').read_text())
    for name,h in build['files'].items():assert sha(parent/name)==h,name
    assert not out.exists(),'Use a new bundle; never modify an existing model'
    shutil.copytree(parent,out,symlinks=True,ignore=shutil.ignore_patterns('plugin-site','cache','BUILD.json','__pycache__','*.pyc'))
    subprocess.run([sys.executable,str(ROOT/'scripts/stage-tool-runtime.py'),
        '--parent-plugin',str(parent/'plugin-site/ornith_g256'),'--native-build',str(a.native_build),
        '--out',str(out/'plugin-site')],check=True)
    preserve_plugin_metadata(parent/'plugin-site', out/'plugin-site')
    subprocess.run(['cp','-a','--reflink=auto',str(parent/'cache'),str(out/'cache')],check=True)
    # New runtime identity, unchanged model entrypoint/settings and all original
    # model/kernel package files. Mutable compilation caches are privately copied.
    allowed={'plugin-site/ornith_g256/__init__.py','plugin-site/ornith_g256/_vllm_correctness/manifest.json'}
    changed=[name for name,h in build['files'].items() if sha(out/name)!=h]
    assert set(changed)<=allowed,changed
    assert sha(out/'plugin-site/ornith_g256/optimized/grammar_parallel.py')=='e9c3608438712521980d7fcd60f58862d0b0a2876620183f304c64a03817edea'
    files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*'))
           if f.is_file() and 'cache' not in f.relative_to(out).parts and '__pycache__' not in f.parts}
    record=dict(status='ASSEMBLED_PENDING_TOOL_INTEGRATION_QUALIFICATION',root=str(out),
        parent_build_sha256=PROMOTED_PARENT,files=files,
        included=build['included']+['R04 native XGrammar finite repetition/shared memory accounting','R03 buffered continuous usage'],
        parent_inclusions_preserved=True,changed_parent_files=changed,
        target_weights=build['target_weights'],draft_weights=build['draft_weights'],weights_modified=False,
        model_file_dependencies=build['model_file_dependencies'],hf_parent_revision=build['hf_parent_revision'],
        tool_integration_sha256=sha(out/'plugin-site/TOOL-INTEGRATION.json'),
        source_script_sha256=sha(__file__),production_changed=False)
    (out/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],root=str(out),build_sha256=sha(out/'BUILD.json'),included=record['included'])),flush=True)


if __name__=='__main__':main()
