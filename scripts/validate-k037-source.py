#!/usr/bin/env python3
"""CPU source parity and reservation checks; no GPU runtime imports or writes."""
import ast, hashlib, importlib.util, io, json, pathlib, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAMMAR = '_xgrammar_native/libxgrammar-candidate.so'
EXPECTED_GRAMMAR = '92ab36bfae20c6c4df3f5321ca375204799ac833252921d9d8f6a206491d35f7'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def normalized(text):
    tree = ast.parse(text.replace('ornith_g256.iu4_correctness.', 'tools.'))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FROZEN_SOURCES' for t in n.targets))]
    return ast.dump(tree, include_attributes=False)
def main():
    assert __debug__, 'Run without -O'
    plugin = ROOT/'plugin-site/ornith_g256'
    pins = json.loads((plugin/'iu4_correctness/source-pins.json').read_text())['files']
    assert len(pins)==102 and pins[GRAMMAR]==EXPECTED_GRAMMAR
    for name,h in pins.items():
        path=plugin/name
        if name==GRAMMAR and not path.exists(): continue
        assert path.is_file() and not path.is_symlink() and sha(path)==h, name
    inputs=json.loads((ROOT/'tests/correctness/port-inputs.json').read_text())
    for name,row in inputs['qualified_input_mapping'].items():
        original=ROOT/row['source'];staged=ROOT/row['staged']
        assert sha(original)==row['source_sha256'], name
        if name == 'adaptive_k3_geometry.py':
            before = "GRAPH_SHA256 = '6f51f0dd740f6546829e46bbbe840df2780ce8c49978658eb3a05092554ff90c'"
            after = "GRAPH_SHA256 = 'fa4ddbb0609ed4ba0065b9eb2753a7a4a9f79edb37cb75ea875973b9fb0eab67'"
            original_text, staged_text = original.read_text(), staged.read_text()
            assert pins['dynamic_graphs.py'] == 'fa4ddbb0609ed4ba0065b9eb2753a7a4a9f79edb37cb75ea875973b9fb0eab67'
            assert original_text.count(before) == staged_text.count(after) == 1
            assert sha(original) == row['staged_sha256']
            assert staged_text.replace(after, before) == original_text
            assert normalized(original_text) == normalized(staged_text.replace(after, before))
        else:
            assert sha(staged)==row['staged_sha256'], name
            assert normalized(original.read_text())==normalized(staged.read_text()), name
    provenance=json.loads((ROOT/'SOURCE-PROVENANCE.json').read_text())
    for name,h in provenance['native_source_files'].items(): assert sha(ROOT/name)==h, name
    for path in plugin.rglob('*.py'): compile(path.read_text(),str(path),'exec')
    path=ROOT/'tests/correctness/test_staged_concurrent_k0.py'
    spec=importlib.util.spec_from_file_location('source_guard_cpu_tests',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert result.wasSuccessful() and result.testsRun==25, stream.getvalue()
    print(json.dumps(dict(passed=True, source_pins=102, source_files_local=101 if not (plugin/GRAMMAR).exists() else 102,
        compiled_grammar_packaged_hf_only=not (plugin/GRAMMAR).exists(), grammar_sha256=EXPECTED_GRAMMAR,
        helper_ports_ast_equal=len(inputs['qualified_input_mapping']), guard_cpu_tests=result.testsRun,
        native_sources_unchanged=True, model_executed=False, publication=False),indent=2))
if __name__=='__main__': main()
