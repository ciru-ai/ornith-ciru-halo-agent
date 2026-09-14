"""Load the four source-only corrections for the pinned Ciru vLLM runtime.

The installed runtime and its compiled libraries are left intact. This finder
only selects versioned Python modules shipped with this integration package.
"""
# Copyright 2026 Ciru. Licensed under Apache-2.0.
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

VERSION = "1.0.1"
_ROOT = Path(__file__).parent / "_vllm_correctness"
_FINDER = None


class _CorrectnessSources(importlib.abc.MetaPathFinder):
    def __init__(self, modules):
        self.modules = modules

    def find_spec(self, fullname, path=None, target=None):
        source = self.modules.get(fullname)
        if source is None:
            return None
        return importlib.util.spec_from_file_location(fullname, source)


def install():
    """Register before vLLM worker/model imports; reject an incompatible runtime."""
    global _FINDER
    if _FINDER is not None:
        return
    manifest = json.loads((_ROOT / "manifest.json").read_text())
    spec = importlib.machinery.PathFinder.find_spec("vllm", sys.path)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Ciru runtime 1.0.1 requires the supplied vLLM installation")
    native_root = Path(next(iter(spec.submodule_search_locations)))
    modules = {}
    for name, entry in manifest["modules"].items():
        source = _ROOT / entry["file"]
        native = native_root / entry["native_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Ciru runtime source integrity check failed: {name}")
        if hashlib.sha256(native.read_bytes()).hexdigest() != entry["native_sha256"]:
            raise RuntimeError(f"Ciru runtime 1.0.1 requires its pinned vLLM source: {name}")
        loaded = sys.modules.get(name)
        if loaded is not None:
            raise RuntimeError(f"Import ornith_g256 before {name} to load Ciru runtime 1.0.1")
        modules[name] = source
    _FINDER = _CorrectnessSources(modules)
    sys.meta_path.insert(0, _FINDER)
