"""Schedule qualified Q/K reductions independently of compiler symbol numbering."""
import ast
import copy
import hashlib
import json
from pathlib import Path

Q = 'triton_red_fused_clone_rms_norm_split_split_with_sizes_view_6'
K = 'triton_red_fused_rms_norm_split_with_sizes_view_7'
Q_HASH = 'e9c97db55aab3cf4fbed316817db375ccb4651d2b323c709aeadbe6f5a38eb36'
K_HASH = 'd8cf836637f3133f2c79f42d717629ca300ce9f7705dd6489f5089099af98a77'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def ast_hash(value):
    return digest(ast.dump(value, include_attributes=False))


def pointer_names(expression):
    return {n.id for n in ast.walk(expression) if isinstance(n, ast.Name)}


def transform(source):
    if 'clone_rms_norm_split_split_with_sizes_view' not in source:
        return source, None, None
    tree = ast.parse(source)
    kernels = {}
    roles = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'triton' and len(node.args) > 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[1], ast.Constant)):
            continue
        name = node.args[0].value
        fn = next(n for n in ast.parse(node.args[1].value).body if isinstance(n, ast.FunctionDef))
        kernels[name] = fn
        normalized = copy.deepcopy(fn)
        normalized.decorator_list = []
        for role, canonical, expected in (('q', Q, Q_HASH), ('k', K, K_HASH)):
            normalized.name = canonical
            if ast_hash(normalized) == expected:
                assert role not in roles, (role, name)
                roles[role] = name
    if not roles:
        return source, None, None
    assert set(roles) == {'q', 'k'}, roles
    q, k = roles['q'], roles['k']
    matches = []
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        calls = {x.value.func.value.id: i for i, x in enumerate(body)
                 if isinstance(x, ast.Expr) and isinstance(x.value, ast.Call)
                 and isinstance(x.value.func, ast.Attribute) and x.value.func.attr == 'run'
                 and isinstance(x.value.func.value, ast.Name)}
        if q in calls and k in calls:
            matches.append((body, calls[q], calls[k]))
    assert len(matches) == 1, len(matches)
    body, i, j = matches[0]
    assert j >= i + 4
    # The released draft graph has already adjacent Q/K reductions (_2/_3).
    # It does not use the target substitution. Preserve that execution path.
    # The target scheduler moves K across Q consumers; cached canonical output
    # is the only already adjacent pair belonging to that target schedule.
    if j == i + 4 and (q, k) != (Q, K):
        return source, None, None
    moved = body[j - 3:j + 1]
    between = body[i + 1:j - 3]
    qcall, kcall = body[i].value, moved[-1].value
    assert len(qcall.args) == len(kcall.args) == 4
    assert ast_hash(qcall.args[0]) == ast_hash(kcall.args[0])
    assert isinstance(qcall.args[0], ast.Name)
    assert isinstance(kcall.args[1], ast.Name)
    assert all(isinstance(c.args[3], ast.Constant) and c.args[3].value == 256 for c in (qcall, kcall))
    assert isinstance(moved[0], ast.Assign) and isinstance(moved[0].value, ast.Call)
    assert moved[0].value.func.id == 'empty_strided_cuda'
    size = moved[0].value.args[0].elts[0]
    assert isinstance(size, ast.Name)
    names = {qcall.args[0].id: 'input', kcall.args[1].id: 'ksum', size.id: 'rows',
             k + '_xnumel': 'kcount', k: 'kreduce'}

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id)
            return node

    def normalized(nodes):
        return Normalize().visit(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))

    expected = ast.parse('ksum = empty_strided_cuda((rows, 2, 1), (2, 1, 2*rows), torch.float32)\n'
                         'kcount = 2*rows\nraw_stream0 = get_raw_stream(0)\n'
                         'kreduce.run(input, ksum, kcount, 256, stream=raw_stream0)').body
    assert ast_hash(normalized(moved)) == ast_hash(ast.Module(body=expected, type_ignores=[]))
    protected = {qcall.args[0].id, kcall.args[1].id, size.id}
    checks = []
    for statement in between:
        assert not any(isinstance(n, ast.Name) and n.id in protected and isinstance(n.ctx, (ast.Store, ast.Del))
                       for n in ast.walk(statement)), ast.unparse(statement)
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if pointer_names(node) & protected:
                    if node.func.id != 'assert_size_stride':
                        assert node.func.id in ('empty_strided_cuda', 'reinterpret_tensor', 'get_raw_stream', 'copy_if_misaligned')
                        assert not pointer_names(node) & (protected - {size.id}), ast.unparse(node)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == 'run':
                assert isinstance(node.func.value, ast.Name) and node.func.value.id in kernels
                fn = kernels[node.func.value.id]
                params = [a.arg for a in fn.args.args]
                stores = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute) and n.func.attr in ('store', 'atomic_add', 'atomic_xchg', 'atomic_cas')]
                written = set().union(*(pointer_names(n.args[0]) for n in stores)) if stores else set()
                for index, param in enumerate(params[:len(node.args)]):
                    if param in written:
                        assert not pointer_names(node.args[index]) & protected, ast.unparse(node)
                checks.append({'kernel': node.func.value.id, 'written_arguments': sorted(written)})
            else:
                assert not pointer_names(node) & protected, ast.unparse(node)
    lines = source.splitlines(keepends=True)
    end_q, begin_k, end_k = body[i].end_lineno, moved[0].lineno - 1, moved[-1].end_lineno
    order = list(range(end_q)) + list(range(begin_k, end_k)) + list(range(end_q, begin_k)) + list(range(end_k, len(lines)))
    reordered = ''.join(lines[n] for n in order)
    inverse = {old: new for new, old in enumerate(order)}
    assert ''.join(reordered.splitlines(keepends=True)[inverse[n]] for n in range(len(lines))) == source
    for old, new in ((q, Q), (k, K)):
        assert old == new or new not in source, (old, new)
        reordered = reordered.replace(old, new)
    body[i + 1:j + 1] = moved + between
    expected_source = ast.unparse(tree).replace(q, Q).replace(k, K)
    assert ast_hash(ast.parse(reordered)) == ast_hash(ast.parse(expected_source))
    return reordered, {old + 1: new + 1 for new, old in enumerate(order)}, {
        'parent_sha256': digest(source), 'candidate_sha256': digest(reordered),
        'q_ast_sha256': Q_HASH, 'k_ast_sha256': K_HASH, 'original_symbols': roles,
        'intervening_statements': len(between), 'intervening_writes_checked': checks,
        'reverse_exact': True, 'arithmetic_unchanged': True,
        'delta': 'Identify exact reduction bodies; move independent K allocation/reduction before Q consumers; canonicalize symbols.'}


def install(output):
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.utils import ValueWithLineMap
    out = Path(output) / 'qk-schedule'
    out.mkdir()
    original = PythonWrapperCodegen.generate
    records = []

    def generate(wrapper, *args, **kwargs):
        result = original(wrapper, *args, **kwargs)
        source = result[0].value
        try:
            modified, mapping, proof = transform(source)
        except Exception:
            (out / 'unqualified-source.py').write_text(source)
            raise
        if proof is None:
            return result
        index = len(records)
        (out / f'{index}-parent.py').write_text(source)
        (out / f'{index}-candidate.py').write_text(modified)
        records.append(proof)
        (out / 'proof.json').write_text(json.dumps(records, indent=2) + '\n')
        line_map = sorted(((mapping.get(line, line), context) for line, context in result[0].line_map), key=lambda item: item[0])
        return (ValueWithLineMap(modified, line_map), *result[1:])

    PythonWrapperCodegen.generate = generate
