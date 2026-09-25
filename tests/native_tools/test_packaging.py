"""Regression checks for shipped source and fail-closed native selection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'patches'))
from stream_usage import patch_source
from receiving_stream_usage import patch_receiving_source


class Packaging(unittest.TestCase):
    def test_receiving_source_adapts_to_exact_qualified_overlay(self):
        source=(ROOT/'tests/native_tools/upstream/baseline/receiving_chat_completion_serving.py').read_text()
        shipped=(ROOT/'plugin-site/ornith_g256/_vllm_correctness/chat_completion_serving_r03.py').read_text()
        self.assertEqual(patch_receiving_source(source),shipped)
        manifest=json.loads((ROOT/'plugin-site/ornith_g256/_vllm_correctness/manifest.json').read_text())
        self.assertEqual(manifest['modules']['vllm.entrypoints.openai.chat_completion.serving']['native_sha256'],hashlib.sha256(source.encode()).hexdigest())
        with self.assertRaises(RuntimeError):patch_receiving_source(source+'\n')

    def test_parent_distribution_entrypoints_survive_composition(self):
        spec=importlib.util.spec_from_file_location('assemble_unified',ROOT/'scripts/assemble-unified.py')
        assembly=importlib.util.module_from_spec(spec);spec.loader.exec_module(assembly)
        parent=self.root/'parent';out=self.root/'out';parent.mkdir();out.mkdir()
        (parent/'ornith_g256').mkdir();(parent/'ornith_g256/__init__.py').write_text('old initializer')
        (out/'ornith_g256').mkdir();(out/'ornith_g256/__init__.py').write_text('composed initializer')
        metadata=parent/'ciru_ornith_g256-1.0.2.dist-info';metadata.mkdir()
        for name,data in {'entry_points.txt':'[vllm.general_plugins]\nornith = ornith_g256:register\n','INSTALLER':'pip\n','METADATA':'Version: 1.0.2\n'}.items():
            (metadata/name).write_text(data)
        (parent/'extra.pth').write_text('existing path hook\n')
        assembly.preserve_plugin_metadata(parent,out)
        for f in metadata.iterdir():self.assertEqual(f.read_bytes(),(out/metadata.name/f.name).read_bytes())
        self.assertEqual((out/'extra.pth').read_bytes(),(parent/'extra.pth').read_bytes())
        self.assertEqual((out/'ornith_g256/__init__.py').read_text(),'composed initializer')

    def test_shipped_overlay_is_exact_qualified_patch(self):
        baseline=(ROOT/'tests/native_tools/upstream/baseline/chat_completion_serving.py').read_text()
        self.assertEqual(patch_source(baseline),(ROOT/'plugin-site/ornith_g256/_vllm_correctness/chat_completion_serving_r03.py').read_text())
        with self.assertRaises(RuntimeError):patch_source(baseline+'\n')

    def test_r01_is_unchanged(self):
        self.assertEqual(hashlib.sha256((ROOT/'plugin-site/ornith_g256/grammar_parallel.py').read_bytes()).hexdigest(),
                         'e9c3608438712521980d7fcd60f58862d0b0a2876620183f304c64a03817edea')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        spec=importlib.util.spec_from_file_location('isolated_native_loader',ROOT/'plugin-site/ornith_g256/xgrammar_native.py')
        self.loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.loader)
        self.loader.__file__=str(self.root/'xgrammar_native.py')
        self.native=self.root/'wheel.so';self.native.write_bytes(b'original fixture')
        self.loader.BASELINE_SHA256=hashlib.sha256(self.native.read_bytes()).hexdigest()
        self.library=self.root/'_xgrammar_native/libxgrammar-candidate.so';self.library.parent.mkdir();self.library.write_bytes(b'candidate fixture')
        self.selection=self.library.parent/'selection.json'
        self.selection.write_text(json.dumps(dict(mode='candidate',sha256=hashlib.sha256(self.library.read_bytes()).hexdigest())))
        native=self.native
        class Distribution:
            version='0.2.3'
            def locate_file(self,name):return native
        self.dist=Distribution()
        self.mock=patch.object(self.loader.importlib.metadata,'distribution',return_value=self.dist);self.mock.start()

    def tearDown(self):
        if self.loader._FINDER in sys.meta_path:sys.meta_path.remove(self.loader._FINDER)
        self.mock.stop();self.tmp.cleanup()

    def test_wrong_wheel_version_rejected(self):
        self.dist.version='0.2.4'
        with self.assertRaises(RuntimeError):self.loader.install()

    def test_changed_installed_library_rejected(self):
        self.native.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):self.loader.install()

    def test_late_install_rejected(self):
        with patch.dict(sys.modules,{'xgrammar.load_binding':object()}):
            with self.assertRaises(RuntimeError):self.loader.install()

    def test_invalid_selection_rejected(self):
        self.selection.write_text(json.dumps(dict(mode='unknown',sha256='x')))
        with self.assertRaises(RuntimeError):self.loader.install()

    def test_install_once_and_verify_at_load(self):
        self.loader.install();finder=self.loader._FINDER;self.loader.install()
        self.assertEqual(sys.meta_path.count(finder),1)
        self.assertIsNone(finder.find_spec('unrelated.module'))
        self.assertEqual(finder.find_spec('xgrammar.load_binding').origin,str(self.library))
        self.library.write_bytes(b'corrupted after installation')
        with self.assertRaises(RuntimeError):finder.exec_module(object())


if __name__=='__main__':unittest.main()
