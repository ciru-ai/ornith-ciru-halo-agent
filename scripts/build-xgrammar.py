#!/usr/bin/env python3
"""Build pinned R04 in a fresh directory, leaving the installed wheel intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', help='Optional local Git repository; its working tree is never edited')
    parser.add_argument('--control', action='store_true', help='Also build the matched unpatched control')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if not args.python.is_file() or args.jobs < 1:
        parser.error('Use an existing receiving-runtime Python and positive job count')
    out = args.out.absolute()
    out.mkdir(parents=True, exist_ok=False)
    identity = json.loads((ROOT / 'patches/xgrammar/SOURCE.json').read_text())
    patch = ROOT / 'patches/xgrammar/r04.patch'
    assert sha(patch) == identity['patch_sha256']
    records = []
    for mode in (['control', 'candidate'] if args.control else ['candidate']):
        source, build = out / ('source-' + mode), out / ('build-' + mode)
        subprocess.run(['git', 'clone', '--no-hardlinks', args.source or identity['repository'], str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', identity['commit']], check=True)
        subprocess.run(['git', '-C', str(source), 'submodule', 'update', '--init', '--recursive'], check=True)
        if mode == 'candidate':
            subprocess.run(['git', '-C', str(source), 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', str(source), 'apply', str(patch)], check=True)
            for name, digest in identity['candidate_source_sha256'].items():
                assert sha(source / name) == digest, name
        configure = ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DXGRAMMAR_BUILD_PYTHON_BINDINGS=ON',
                     '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -fno-lto',
                     '-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O3 -g -DNDEBUG -fno-lto',
                     '-DPython_EXECUTABLE=' + str(args.python), '-DPython3_EXECUTABLE=' + str(args.python)]
        started = time.monotonic()
        subprocess.run(configure, check=True)
        subprocess.run(['cmake', '--build', str(build), '--parallel', str(args.jobs)], check=True)
        library = out / ('libxgrammar-' + mode + '.so')
        shutil.copy2(source / 'python/xgrammar/libxgrammar_bindings.so', library)
        records.append(dict(mode=mode,library=library.name,sha256=sha(library),
            source_commit=identity['commit'],patch_sha256=identity['patch_sha256'] if mode=='candidate' else None,
            sources={n:sha(source/n) for n in identity['candidate_source_sha256']},
            configure=configure,build_seconds=time.monotonic()-started,
            compiler=subprocess.check_output([os.environ.get('CXX','c++'),'--version'],text=True),
            submodules=subprocess.check_output(['git','-C',str(source),'submodule','status','--recursive'],text=True)))
        (out/'build-receipt.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(status='BUILT_PENDING_RECEIVING_QUALIFICATION',builds=records)),flush=True)


if __name__ == '__main__':
    main()
