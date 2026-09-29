import os, ast, ctypes as C, importlib.util, json, sys, types
from pathlib import Path
import torch, triton

ROOT=Path(os.environ['CIRU_SCREEN_ROOT']); SOURCE=ROOT/'source/plugin-site/ornith_g256'
torch.manual_seed(617);results={}
def module(name):
    spec=importlib.util.spec_from_file_location('screen_'+name,SOURCE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
gemv=module('small_gemv');topk=module('fast_topk')
rows=[]
for m,n,k in [(1,256,2048),(4,256,2048),(4,4096,2048),(8,64,4096)]:
    x=torch.randn((m,k),device='cuda').bfloat16();w=torch.randn((n,k),device='cuda').bfloat16()
    got=gemv.small_gemm(x,w,None,False);expected=torch.nn.functional.linear(x,w)
    rms=((got.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
    assert rms<.004,rms;rows.append({'m':m,'n':n,'k':k,'relative_rms':rms})
results['bf16_gemv']=rows
for renorm in (True,False):
    logits=torch.randn((16,256),device='cuda')*3;weights=torch.empty((16,8),device='cuda');ids=torch.empty((16,8),dtype=torch.int32,device='cuda')
    topk._topk_softmax[(16,)](logits,weights,ids,256,E=256,TOPK=8,RENORM=renorm,num_warps=4)
    refw,refids=logits.softmax(-1).topk(8,-1)
    if renorm:refw=refw/refw.sum(-1,keepdim=True)
    assert torch.equal(ids,refids.int());assert torch.allclose(weights,refw,rtol=2e-6,atol=1e-7)
results['router']='PASS_32_rows'
tree=ast.parse((SOURCE/'_vllm_correctness/qwen_gdn_linear_attn.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name in ('triton','triton.language') for a in n.names) or isinstance(n,ast.FunctionDef) and n.name in ('_gdn_state_copy_kernel','_copy_state_rows')]
copyfile=ROOT/'gdn_copy_screen_module.py';copyfile.write_text('import torch\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[])))
spec=importlib.util.spec_from_file_location('gdn_copy_screen_module',copyfile);copy=importlib.util.module_from_spec(spec);spec.loader.exec_module(copy)
for dtype in (torch.bfloat16,torch.float32):
    for src,dst in [(0,3),(2,2)]:
        state=torch.randn((5,16,128,128),device='cuda',dtype=dtype);expected=state.clone();s=torch.tensor([src],device='cuda');d=torch.tensor([dst],device='cuda')
        expected[d]=expected[s];copy._copy_state_rows(state,s,d);assert torch.equal(state,expected)
results['gdn_copy']='PASS_4_exact_cases'
native=module('native');routed=ROOT/'ornith-baseline/bundle/native/libornith_routed_storage_n32.so'
old=C.CDLL(str(routed));new=C.CDLL(str(ROOT/'native-built/libornith_routed_storage_n32.so'))
for lib,symbol in ((old,'ornith_routed_direct_get_layout'),(new,'ornith_routed_shared_get_layout')):
    q=getattr(lib,symbol);q.argtypes=[C.c_int,C.POINTER(native.RoutedLayout)];q.restype=C.c_int
old.ornith_routed_direct_launch.argtypes=[C.c_void_p]*8+[C.c_size_t]+[C.c_void_p]*2+[C.c_int]*2+[C.c_void_p]
new.ornith_routed_direct_launch_shared.argtypes=[C.c_void_p]*13+[C.c_size_t]+[C.c_void_p]*3+[C.c_int]*2+[C.c_void_p]
gate=torch.randint(-(2**31),2**31-1,(256,1024//32,2048//256,32,32),device='cuda',dtype=torch.int32)
down=torch.randint(-(2**31),2**31-1,(256,2048//32,512//256,32,32),device='cuda',dtype=torch.int32)
def metadata(n,k):
    a=torch.full((256,n//32,k//256,32),.001,device='cuda',dtype=torch.float16);b=torch.full_like(a,-.007)
    return (a.view(torch.int16).int()&65535)|(b.view(torch.int16).int()<<16)
gm=metadata(1024,2048);dm=metadata(2048,512)
x=torch.randn((1,2048),device='cuda').bfloat16();weights=torch.zeros((1,8),device='cuda');weights[0,0]=1
refids=torch.full((1,8),-1,dtype=torch.int32,device='cuda');refids[0,0]=0
oldlayout=native.RoutedLayout();native.check(old.ornith_routed_direct_get_layout(1,C.byref(oldlayout)),'old layout')
oldwork=torch.empty(oldlayout.workspace_bytes,device='cuda',dtype=torch.uint8);reference=torch.empty_like(x);flags=torch.zeros(1,dtype=torch.int32,device='cuda')
native.check(old.ornith_routed_direct_launch(*[native.ptr(t) for t in (x,weights,refids,gate,gm,down,dm,oldwork)],oldwork.numel(),native.ptr(reference),native.ptr(flags),1,1,native.stream(x)),'old routed')
layout=native.RoutedLayout();native.check(new.ornith_routed_shared_get_layout(1,C.byref(layout)),'shared layout');work=torch.empty(layout.workspace_bytes,device='cuda',dtype=torch.uint8)
ids=torch.full_like(refids,-1);main=torch.empty_like(x);shared=torch.empty_like(x);logit=torch.zeros((1,1),dtype=torch.bfloat16,device='cuda')
native.check(new.ornith_routed_direct_launch_shared(*[native.ptr(t) for t in (x,weights,ids,gate,gm,down,dm,gate[0],gm[0],down[0],dm[0],logit,work)],work.numel(),native.ptr(main),native.ptr(shared),native.ptr(flags),1,1,native.stream(x)),'shared with dead routed slots')
torch.cuda.synchronize();assert flags.item()==0;assert torch.count_nonzero(main).item()==0;assert torch.equal(shared,reference*.5)
results['shared_dead_routes']='PASS_exact_nonzero_shared_output'
(ROOT/'plugin-screen.json').write_text(json.dumps({'status':'PASS','results':results},indent=2)+'\n');print(json.dumps(results,indent=2))
