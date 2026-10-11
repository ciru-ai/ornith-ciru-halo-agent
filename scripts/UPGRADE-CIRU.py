#!/usr/bin/env python3
"""Upgrade an existing Ciru model directory without downloading its weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

REPOS = {
    'ornith': 'jcbtc/Ornith1.5-Ciru-Halo-Agent-vllm-strix-halo',
    'apodex': 'jcbtc/Apodex1.1-mini-Ciru-Halo-Agent-vllm-strix-halo',
}

def control_file(name, include_engine=False):
    if name.startswith('bundle/'):
        return not name.startswith(('bundle/models/', 'bundle/cache/'))
    if name.startswith('runtime/'):
        return include_engine or name.endswith(('.py', '.sh', '.json', '.lock', '.nix')) or '/ciru-v4-' in name
    return name in ('README.md', 'INSTALL.md', 'RELEASE.json', 'SOURCE-PROVENANCE.json', 'RUNTIME-FIXES.md', 'CREDITS.md', '.gitattributes', 'PATCH-NOTES-4.1.3.md') or name.startswith(('evaluation/', 'assets/'))

def verify_selected_controls(payload, release, include_engine=False):
    """Refuse an incomplete selected payload before moving the installed bundle."""
    hashes = release['payload_sha256']
    required = ('bundle/paths.env',
                'bundle/plugin-site/ciru_ornith_g256-4.1.3.dist-info/METADATA')
    for name in required:
        if name not in hashes:
            raise RuntimeError(f'Required release control hash is missing: {name}')
    for name, expected in hashes.items():
        if name == 'RELEASE.json' or not control_file(name, include_engine):
            continue
        path = payload/name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f'Selected release control file is missing: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Release checksum mismatch: {name}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=REPOS, required=True)
    parser.add_argument('--directory', type=Path, required=True, help='Existing model download directory')
    parser.add_argument('--revision', default='main', help='Hub revision; resolves to a fixed commit before download')
    parser.add_argument('--install-runtime', type=Path, help='Install the pinned engine into this NEW directory when upgrading from a pre-v4 engine')
    parser.add_argument('--prepared-directory', type=Path, help='Use a verified offline release folder instead of downloading')
    args = parser.parse_args()
    destination = args.directory.expanduser().resolve()
    bundle = destination/'bundle'
    if not (bundle/'models').exists():
        raise RuntimeError('Expected an existing installation with bundle/models. For a new installation, use the complete download in INSTALL.md.')
    previous = destination/'RELEASE.json'
    previous_version = str(json.loads(previous.read_text()).get('version', 'unknown')) if previous.is_file() else 'unknown'
    if not previous_version.startswith('4.') and not args.install_runtime:
        raise RuntimeError('For pre-v4 or unidentified engines, add --install-runtime NEW_DIRECTORY to install the pinned engine without overwriting your old runtime.')
    # Linux serving processes include model/native paths in their command line.
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        if proc.parent.name == str(os.getpid()):
            continue
        try:
            command = proc.read_bytes()
        except (OSError, PermissionError):
            continue
        if str(bundle).encode() in command and (b'python' in command or b'llama-server' in command):
            raise RuntimeError('Stop the model process before updating its bundle.')
    temporary = Path(tempfile.mkdtemp(prefix='.ciru-upgrade-', dir=destination))
    backup = None
    moved = []
    created_files = []
    try:
        payload = temporary/'payload'
        if args.prepared_directory:
            source = args.prepared_directory.expanduser().resolve()
            for p in source.rglob('*'):
                rel = p.relative_to(source).as_posix()
                if p.is_file() and not p.is_symlink() and control_file(rel, bool(args.install_runtime)) and '.cache' not in p.parts and '__pycache__' not in p.parts:
                    target = payload/rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
            revision = 'verified-offline-folder'
        else:
            from huggingface_hub import HfApi, snapshot_download
            info = HfApi().model_info(REPOS[args.model], revision=args.revision)
            revision = info.sha
            names = [f.rfilename for f in info.siblings if control_file(f.rfilename, bool(args.install_runtime))]
            snapshot_download(REPOS[args.model], revision=revision, local_dir=payload, allow_patterns=names)
        release = json.loads((payload/'RELEASE.json').read_text())
        if release['version'] != '4.1.3':
            raise RuntimeError('This updater requires the 4.1.3 release; no installation files changed.')
        if not args.prepared_directory and not release.get('publication_ready'):
            raise RuntimeError('This release is not ready for installation; no installation files changed.')
        verify_selected_controls(payload, release, bool(args.install_runtime))
        checked = 0
        for p in payload.rglob('*'):
            if p.is_file() and '.cache' not in p.parts:
                rel = p.relative_to(payload).as_posix()
                if rel == 'RELEASE.json':
                    continue
                expected = release['payload_sha256'].get(rel)
                if expected is None or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Release checksum mismatch: {rel}')
                checked += 1
        if not (payload/'bundle/plugin-site/ciru_ornith_g256-4.1.3.dist-info/METADATA').is_file():
            raise RuntimeError('New distribution metadata is missing.')
        if args.install_runtime:
            runtime = args.install_runtime.expanduser().resolve()
            if runtime.exists():
                raise RuntimeError('Choose a NEW runtime directory; existing runtimes are preserved.')
            installer = payload/'runtime'/('INSTALL-ORNITH-RUNTIME.sh' if args.model=='ornith' else 'INSTALL-APODEX-RUNTIME.sh')
            subprocess.run(['bash',str(installer),str(runtime)],check=True)
        backup = destination/'.ciru-upgrade-backups'/f'4.1.3-{time.time_ns()}'
        backup.mkdir(parents=True)
        os.replace(bundle, backup/'bundle')
        for name in ('models','cache'):
            old = backup/'bundle'/name
            if old.exists() or old.is_symlink():
                os.replace(old, payload/'bundle'/name)
                moved.append(name)
        os.replace(payload/'bundle', bundle)
        for p in payload.rglob('*'):
            if p.is_file() and '.cache' not in p.parts:
                rel = p.relative_to(payload)
                target = destination/rel
                if target.is_file():
                    saved = backup/rel
                    saved.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(target,saved)
                elif target.exists():
                    raise RuntimeError(f'Expected a file at {target}; preserved the installation.')
                else:
                    created_files.append(target)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,target)
        receipt = dict(version='4.1.3', repo=REPOS[args.model], revision=revision, verified_files=checked, backup=str(backup), weights_downloaded=False, weights_preserved=True, compile_cache_preserved=True)
        (destination/'UPGRADE-RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2))
        if args.install_runtime:
            variable='ORNITH_RUNTIME_ROOT' if args.model=='ornith' else 'APODEX_RUNTIME_ROOT'
            print(f'Use {variable}={runtime} with your usual bundle/serve.sh command.')
        else:
            print('Ciru v4 users: reuse your installed pinned engine and restart the usual launcher. Pre-v4 users: follow the new-runtime step in INSTALL.md before starting.')
    except Exception:
        if backup is not None and (backup/'bundle').exists():
            if bundle.exists():
                for name in moved:
                    if (bundle/name).exists() or (bundle/name).is_symlink():
                        os.replace(bundle/name,backup/'bundle'/name)
                shutil.rmtree(bundle)
            else:
                for name in moved:
                    os.replace(payload/'bundle'/name,backup/'bundle'/name)
            os.replace(backup/'bundle',bundle)
            for p in created_files:
                if p.is_file():
                    p.unlink()
            # Restore any top-level files replaced before an interrupted copy.
            for p in backup.rglob('*'):
                if p.is_file():
                    target=destination/p.relative_to(backup)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(p,target)
        raise
    finally:
        shutil.rmtree(temporary)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # Do not print authenticated HTTP diagnostics or credential values.
        print(f'Upgrade failed: {type(exc).__name__}. For local validation errors, {str(exc) if isinstance(exc, RuntimeError) else "check INSTALL.md and retry"}')
        raise SystemExit(1)
