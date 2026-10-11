#!/usr/bin/env python3
"""Offline updater preservation/refusal/rollback checks on tiny installations.

Only updater-selected control files are copied from the prepared release. Model
and compile-cache sentinels are private, tiny files; the release is never edited.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    spec = importlib.util.spec_from_file_location('proposed_updater_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controls(module, root):
    return sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
        and module.control_file(p.relative_to(root).as_posix(), False)
        and '.cache' not in p.parts and '__pycache__' not in p.parts)


def _copy_controls(module, source, destination):
    for source_file in _controls(module, source):
        target = destination / source_file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
    return destination


def _fixture(root, version='4.1.2'):
    files = {
        'RELEASE.json': json.dumps({'version': version})+'\n',
        'README.md': 'Original installation readme\n',
        'INSTALL.md': 'Original installation instructions\n',
        'SOURCE-PROVENANCE.json': '{"old": true}\n',
        'bundle/paths.env': 'OLD_LOCAL_PATHS=preserved-until-upgrade\n',
        'bundle/serve.sh': '#!/bin/sh\n# original launcher\n',
        'bundle/plugin-site/ornith_g256/__init__.py': '# old plugin\n',
        'bundle/plugin-site/ciru_ornith_g256-4.1.2.dist-info/METADATA': 'Name: ciru-ornith-g256\nVersion: 4.1.2\n',
        'bundle/models/target/model.safetensors': 'private tiny target weight sentinel\n',
        'bundle/models/draft/model.safetensors': 'private tiny draft weight sentinel\n',
        'bundle/cache/compiled.bin': 'private compile cache sentinel\n',
        'runtime/local-control.py': '# original runtime control\n',
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def _snapshot(root, include_directories=False):
    result = {}
    for path in root.rglob('*'):
        if path.is_file():
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = {'sha256': _sha(path), 'inode': stat.st_ino,
                                                       'device': stat.st_dev, 'mode': stat.st_mode}
        elif include_directories and path.is_dir():
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = {'inode': stat.st_ino,
                                                       'device': stat.st_dev, 'mode': stat.st_mode}
    return result


def _unchanged(root, before):
    for name, expected in before.items():
        path = root / name
        assert path.exists(), 'Original path lost: '+name
        stat = path.stat()
        actual = {'inode': stat.st_ino, 'device': stat.st_dev, 'mode': stat.st_mode}
        if 'sha256' in expected:
            assert path.is_file(), 'Original file replaced by non-file: '+name
            actual['sha256'] = _sha(path)
        assert actual == expected, 'Original bytes or inode changed: '+name


def _invoke(module, installed, prepared):
    arguments = ['UPGRADE-CIRU.py', '--model', 'ornith', '--directory', str(installed),
                 '--prepared-directory', str(prepared)]
    output = io.StringIO()
    with patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(output):
        module.main()
    return output.getvalue()


def _refusal(module, installed, prepared):
    before = _snapshot(installed, True)
    try:
        _invoke(module, installed, prepared)
    except RuntimeError as error:
        _unchanged(installed, before)
        assert not (installed / 'UPGRADE-RECEIPT.json').exists()
        return {'refused': True, 'reason': str(error), 'original_paths_preserved': len(before)}
    raise AssertionError('Invalid upgrade was accepted')


def run_checks(updater_path, prepared_release):
    updater_path, prepared_release = Path(updater_path).resolve(), Path(prepared_release).resolve()
    module = _load(updater_path)
    release = json.loads((prepared_release / 'RELEASE.json').read_text())
    assert release['version'] == '4.1.4'
    assert _sha(prepared_release / 'runtime/UPGRADE-CIRU.py') == _sha(updater_path), 'Prepared updater differs from the tested script'
    controls = _controls(module, prepared_release)
    names = [p.relative_to(prepared_release).as_posix() for p in controls]
    assert 'bundle/paths.env' in names
    assert 'bundle/plugin-site/ciru_ornith_g256-4.1.4.dist-info/METADATA' in names
    for path, name in zip(controls, names):
        if name != 'RELEASE.json':
            assert release['payload_sha256'].get(name) == _sha(path), 'Prepared control hash mismatch: '+name
    receipt = {'schema': 'ciru.ornith.proposed_updater.cpu_checks.v2',
        'updater_sha256': _sha(updater_path), 'prepared_release_sha256': _sha(prepared_release / 'RELEASE.json'),
        'control_files_selected': len(names), 'expected_verified_files': len(names)-1,
        'prepared_updater_byte_identical_to_tested_script': True,
        'models_or_cache_copied_from_release': False, 'offline_only': True,
        'source_release_modified': False, 'live_installation_touched': False,
        'publication_or_runtime_qualification': False, 'checks': []}
    with tempfile.TemporaryDirectory(prefix='ciru-proposed-updater-check-') as temporary:
        root = Path(temporary)
        prepared = _copy_controls(module, prepared_release, root / 'prepared-controls')

        def check(name, action):
            try:
                details = action()
                receipt['checks'].append({'name': name, 'status': 'PASS', **details})
            except Exception as error:
                receipt['checks'].append({'name': name, 'status': 'FAIL', 'error': type(error).__name__+': '+str(error)})

        def success():
            installed = _fixture(root / 'success')
            preserve = {name: value for name, value in _snapshot(installed, True).items()
                        if name.startswith(('bundle/models', 'bundle/cache'))}
            _invoke(module, installed, prepared)
            _unchanged(installed, preserve)
            result = json.loads((installed / 'UPGRADE-RECEIPT.json').read_text())
            assert result['verified_files'] == len(names)-1
            assert result['version'] == '4.1.4' and result['weights_preserved'] and result['compile_cache_preserved']
            assert not result['weights_downloaded']
            distributions = sorted(p.name for p in (installed / 'bundle/plugin-site').glob('ciru_ornith_g256-*.dist-info'))
            assert distributions == ['ciru_ornith_g256-4.1.4.dist-info'], distributions
            return {'verified_files': result['verified_files'], 'weight_and_cache_paths_preserved': len(preserve),
                    'new_distribution_only': True, 'sha_and_inode_preservation': True}
        check('offline_complete_control_upgrade_preserves_weights_and_cache', success)

        def tampered():
            damaged = _copy_controls(module, prepared, root / 'tampered-controls')
            path = damaged / 'bundle/paths.env'
            path.write_bytes(path.read_bytes()+b'# checksum tamper\n')
            return _refusal(module, _fixture(root / 'tampered-install'), damaged)
        check('tampered_checksum_refused_without_installation_changes', tampered)

        def missing_bundle():
            damaged = _copy_controls(module, prepared, root / 'missing-bundle-controls')
            shutil.rmtree(damaged / 'bundle')
            return _refusal(module, _fixture(root / 'missing-bundle-install'), damaged)
        check('missing_prepared_bundle_refused_without_installation_changes', missing_bundle)

        def missing_hash():
            damaged = _copy_controls(module, prepared, root / 'missing-hash-controls')
            manifest = json.loads((damaged / 'RELEASE.json').read_text())
            del manifest['payload_sha256']['bundle/paths.env']
            (damaged / 'RELEASE.json').write_text(json.dumps(manifest)+'\n')
            return _refusal(module, _fixture(root / 'missing-hash-install'), damaged)
        check('missing_paths_env_hash_refused_without_installation_changes', missing_hash)

        def missing_paths_file():
            damaged = _copy_controls(module, prepared, root / 'missing-paths-file-controls')
            (damaged / 'bundle/paths.env').unlink()
            result = _refusal(module, _fixture(root / 'missing-paths-file-install'), damaged)
            return dict(result, missing_selected_file='bundle/paths.env', manifest_hash_retained=True)
        check('missing_paths_env_file_refused_without_installation_changes', missing_paths_file)

        def missing_native_file():
            damaged = _copy_controls(module, prepared, root / 'missing-native-file-controls')
            candidates = sorted(name for name in names if name.startswith('bundle/native/') and name.endswith('.so'))
            name = candidates[0] if candidates else 'bundle/native/updater-absence-fixture.so'
            path = damaged / name
            if not candidates:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'private selected native sentinel; never loaded\n')
                manifest = json.loads((damaged / 'RELEASE.json').read_text())
                manifest['payload_sha256'][name] = _sha(path)
                (damaged / 'RELEASE.json').write_text(json.dumps(manifest)+'\n')
            path.unlink()
            result = _refusal(module, _fixture(root / 'missing-native-file-install'), damaged)
            return dict(result, missing_selected_file=name, manifest_hash_retained=True,
                        native_library_loaded=False)
        check('missing_selected_native_file_refused_without_installation_changes', missing_native_file)

        check('pre_v4_without_new_runtime_refused_without_installation_changes',
              lambda: _refusal(module, _fixture(root / 'pre-v4-install', '3.9.0'), prepared))

        def rollback():
            installed = _fixture(root / 'rollback-install')
            before = _snapshot(installed, True)
            original_copy = module.shutil.copy2
            injected = []
            def interrupted_copy(source, destination, *args, **kwargs):
                if (not injected and Path(destination) == installed / 'README.md'
                        and (installed / 'bundle/plugin-site/ciru_ornith_g256-4.1.4.dist-info').is_dir()):
                    injected.append(True)
                    raise OSError('Injected top-level copy failure after bundle replacement')
                return original_copy(source, destination, *args, **kwargs)
            with patch.object(module.shutil, 'copy2', interrupted_copy):
                try:
                    _invoke(module, installed, prepared)
                except OSError:
                    pass
                else:
                    raise AssertionError('Injected replacement failure did not propagate')
            assert injected == [True]
            _unchanged(installed, before)
            assert not (installed / 'UPGRADE-RECEIPT.json').exists()
            assert not (installed / 'bundle/plugin-site/ciru_ornith_g256-4.1.4.dist-info').exists()
            return {'injected_after_bundle_replacement': True, 'all_original_paths_preserved': len(before),
                    'all_original_file_hashes_and_inodes_preserved': True}
        check('post_replacement_failure_rolls_back_complete_old_installation', rollback)
    receipt['passed'] = all(item['status'] == 'PASS' for item in receipt['checks'])
    receipt['status'] = 'PROPOSED_UPDATER_OFFLINE_CPU_CHECKS_PASS' if receipt['passed'] else 'PROPOSED_UPDATER_OFFLINE_CPU_CHECKS_FAILED'
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--updater', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = run_checks(args.updater, args.prepared)
    text = json.dumps(result, indent=2)+'\n'
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end='')
    raise SystemExit(0 if result['passed'] else 1)
