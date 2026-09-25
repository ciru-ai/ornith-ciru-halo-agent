"""The optional native grammar payload makes the release wheel platform-specific."""
from pathlib import Path
import hashlib
import json
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class NativeWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def run(self):
        root=Path(__file__).parent/'plugin-site/ornith_g256/_xgrammar_native'
        if not (root/'selection.json').is_file() or not (root/'libxgrammar-candidate.so').is_file():
            raise RuntimeError('Build and stage the pinned XGrammar library before creating a release wheel')
        selected=json.loads((root/'selection.json').read_text())
        receipt=json.loads((root/'build-receipt.json').read_text())
        digest=hashlib.sha256((root/'libxgrammar-candidate.so').read_bytes()).hexdigest()
        if selected.get('mode')!='candidate' or selected.get('sha256')!=digest or receipt.get('sha256')!=digest:
            raise RuntimeError('Native wheel payload does not match its selection/build receipt')
        super().run()

    def get_tag(self):
        _,_,platform=super().get_tag()
        return 'py3','none',platform


setup(cmdclass={'bdist_wheel':NativeWheel})
