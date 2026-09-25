"""Load the pinned native grammar implementation before XGrammar initializes."""
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

BASELINE_SHA256 = '946bab66b9ad843758ade3cf45a49df5897ce225e70a5f9b4801d6927ac11890'
_FINDER = None

class _NativeBinding(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, library, expected):
        self.library = library
        self.expected = expected
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'xgrammar.load_binding':
            return importlib.util.spec_from_loader(fullname, self, origin=str(self.library))
    def create_module(self, spec): return None
    def exec_module(self, module):
        if hashlib.sha256(self.library.read_bytes()).hexdigest() != self.expected:
            raise RuntimeError('Ciru native grammar library integrity check failed')
        from tvm_ffi.module import load_module
        module.LIB = load_module(str(self.library), keep_module_alive=True)
        module.CIRU_NATIVE_SHA256 = self.expected

def install():
    global _FINDER
    if _FINDER is not None: return
    if 'xgrammar.load_binding' in sys.modules:
        raise RuntimeError('Install the Ciru native grammar loader before importing XGrammar')
    dist = importlib.metadata.distribution('xgrammar')
    native = Path(dist.locate_file('xgrammar/libxgrammar_bindings.so'))
    if dist.version != '0.2.3' or hashlib.sha256(native.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise RuntimeError('Ciru R04 requires its tested XGrammar 0.2.3 binary and Python API')
    root = Path(__file__).parent / '_xgrammar_native'
    selection = json.loads((root / 'selection.json').read_text())
    if selection['mode'] not in ('control', 'candidate'):
        raise RuntimeError('Unknown native grammar selection')
    library = root / ('libxgrammar-' + selection['mode'] + '.so')
    _FINDER = _NativeBinding(library, selection['sha256'])
    sys.meta_path.insert(0, _FINDER)
