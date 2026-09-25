"""Move a proven independent K reduction beside Q before AOT serialization."""
import ast,copy,hashlib,json
from pathlib import Path

Q='triton_red_fused_clone_rms_norm_split_split_with_sizes_view_6'
K='triton_red_fused_rms_norm_split_with_sizes_view_9'
CANONICAL_K='triton_red_fused_rms_norm_split_with_sizes_view_7'
Q_HASH='e9c97db55aab3cf4fbed316817db375ccb4651d2b323c709aeadbe6f5a38eb36'
K_HASH='d8cf836637f3133f2c79f42d717629ca300ce9f7705dd6489f5089099af98a77'
BETWEEN_HASH='b78bcebde0f5fdfe9f9ef18fa0a78457001d3ed0a48121c9baac2ea07dc65517'
def digest(value):return hashlib.sha256(value.encode()).hexdigest()
def ast_hash(value):return digest(ast.dump(value,include_attributes=False))

def transform(source):
    if Q not in source or K not in source:return source,None,None
    assert CANONICAL_K not in source
    tree=ast.parse(source);hashes={};matches=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and node.func.attr=='triton' and isinstance(node.args[0],ast.Constant)
                and node.args[0].value in (Q,K)):
            fn=next(n for n in ast.parse(node.args[1].value).body if isinstance(n,ast.FunctionDef))
            fn.decorator_list=[]
            if fn.name==K:fn.name=CANONICAL_K
            hashes[node.args[0].value]=ast_hash(fn)
        body=getattr(node,'body',None)
        if not isinstance(body,list):continue
        calls={x.value.func.value.id:i for i,x in enumerate(body)
            if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call)
            and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='run'
            and isinstance(x.value.func.value,ast.Name)}
        if Q in calls and K in calls:matches.append((body,calls[Q],calls[K]))
    assert hashes=={Q:Q_HASH,K:K_HASH},hashes
    assert len(matches)==1,len(matches)
    body,i,j=matches[0];assert j-i==17
    between=body[i+1:j-3]
    moved=body[j-3:j+1]
    # Equivalent layer partitions use different buffer/input/symbol names.
    # Normalize names by their roles, then require the exact same statement AST.
    names={}
    def bind(node,role):
        assert isinstance(node,ast.Name)
        assert node.id not in names or names[node.id]==role
        names[node.id]=role
    bind(body[i].value.args[0],'buf16');bind(body[i].value.args[1],'buf23')
    bind(moved[-1].value.args[1],'buf29')
    bind(moved[0].value.args[0].elts[0],'s18')
    bind(between[0].value.args[0],'arg17_1')
    bind(between[2].targets[0],'buf28');bind(between[3].targets[0],'buf26')
    bind(between[6].value.args[3],'buf24');bind(between[6].value.args[4],'buf25')
    bind(between[7].targets[0],'buf27')
    assert len(set(names.values()))==len(names)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,node):
            node.id=names.get(node.id,node.id);return node
    def normalized(nodes):return Normalize().visit(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]))
    assert ast_hash(normalized(between))==BETWEEN_HASH,ast.unparse(normalized(between))
    expected=ast.parse('buf29 = empty_strided_cuda((s18, 2, 1), (2, 1, 2*s18), torch.float32)\n'
        +K+'_xnumel = 2*s18\nraw_stream0 = get_raw_stream(0)\n'
        +K+'.run(buf16, buf29, '+K+'_xnumel, 256, stream=raw_stream0)').body
    assert ast_hash(normalized(moved))==ast_hash(ast.Module(body=expected,type_ignores=[]))
    # Known intervening operations only read buf16/buf23. Their output aliases
    # belong to freshly allocated buf28, while K writes freshly allocated buf29.
    assert not any(isinstance(n,ast.Name) and n.id in ('buf16','buf29')
                   and isinstance(n.ctx,(ast.Store,ast.Del)) for n in ast.walk(normalized(between)))
    lines=source.splitlines(keepends=True)
    qend=body[i].end_lineno;begin=moved[0].lineno-1;end=moved[-1].end_lineno
    order=list(range(qend))+list(range(begin,end))+list(range(qend,begin))+list(range(end,len(lines)))
    reordered=''.join(lines[n] for n in order)
    inverse={old:new for new,old in enumerate(order)}
    assert ''.join(reordered.splitlines(keepends=True)[inverse[n]] for n in range(len(lines)))==source
    final=reordered.replace(K,CANONICAL_K)
    body[i+1:j+1]=moved+between
    expected_tree=ast.parse(ast.unparse(tree).replace(K,CANONICAL_K))
    assert ast_hash(ast.parse(final))==ast_hash(expected_tree)
    return final,{old+1:new+1 for new,old in enumerate(order)},dict(
        parent_sha256=digest(source),candidate_sha256=digest(final),q_ast_sha256=Q_HASH,
        k_ast_sha256=K_HASH,intervening_ast_sha256=BETWEEN_HASH,reverse_exact=True,
        arithmetic_unchanged=True,buffer_roles=names,
        delta='Move independent K allocation/reduction before Q consumers; canonical K symbol')

def install(output):
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.utils import ValueWithLineMap
    out=Path(output)/'qk-schedule';out.mkdir()
    original=PythonWrapperCodegen.generate
    records=[]
    def generate(wrapper,*args,**kwargs):
        result=original(wrapper,*args,**kwargs)
        source=result[0].value
        try:modified,mapping,proof=transform(source)
        except Exception:
            (out/'unqualified-source.py').write_text(source)
            raise
        if proof is None:return result
        index=len(records)
        (out/f'{index}-parent.py').write_text(source)
        (out/f'{index}-candidate.py').write_text(modified)
        records.append(proof);(out/'proof.json').write_text(json.dumps(records,indent=2)+'\n')
        line_map=sorted(((mapping.get(line,line),context) for line,context in result[0].line_map),key=lambda item:item[0])
        return (ValueWithLineMap(modified,line_map),*result[1:])
    PythonWrapperCodegen.generate=generate
