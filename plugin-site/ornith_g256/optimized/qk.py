"""Experimental single-launch Q/K replacement; no installed-source mutation."""
import ast,collections,hashlib,importlib.util,json,threading
from pathlib import Path
import torch,triton
R=Path(__file__).resolve().parents[1]
SOURCE_SHA='17b18769ae84c2e9707a0257181e5c314e9774fc3b4796f8e6787a8302cb5e1f'
Q_AST='e9c97db55aab3cf4fbed316817db375ccb4651d2b323c709aeadbe6f5a38eb36'
K_AST='d8cf836637f3133f2c79f42d717629ca300ce9f7705dd6489f5089099af98a77'
COMBINED_AST='c8a64f66af264ed1645c996eae3b912dc4dcb6e91034ee2df7ef081f22592f05'
CONFIG=dict(XBLOCK=64,R0_BLOCK=64,K_XBLOCK=8,K_RBLOCK=256,num_warps=16,num_stages=1)
_installed=False;_state=None;_out=None

def persist():
 if _state is None:return
 with (_out/'kernel-provenance.json').open('w') as f:json.dump(_state,f,indent=2)

def install(output):
 global _installed,_state,_out
 assert not _installed
 _out=Path(output);_out.mkdir(parents=True,exist_ok=True)
 p=Path(__file__).with_name('mixed_qk_sum.py')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==SOURCE_SHA
 spec=importlib.util.spec_from_file_location('s8x_mixed_qk_sum',p)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);replacement=module.mixed_qk_sum
 def launch(inputs,stream):
  assert len(inputs)==5 and inputs[3]==8*inputs[4]
  assert inputs[0].shape[-1]==9216
  assert stream==torch.cuda.current_stream().cuda_stream,'Explicit and active streams differ'
  grid=(triton.cdiv(inputs[3],64)+triton.cdiv(inputs[4],8),)
  replacement[grid](*inputs,**CONFIG,enable_fp_fusion=True)
 # Compile deterministically before model graph capture, with no RNG use.
 x=torch.zeros((1,9216),device='cuda',dtype=torch.bfloat16)
 q=torch.full((1,16),float('nan'),device='cuda');k=torch.full((1,2),float('nan'),device='cuda')
 launch((x,q,k,16,2),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
 assert torch.count_nonzero(q)==torch.count_nonzero(k)==0
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):launch((x,q,k,16,2),torch.cuda.current_stream().cuda_stream)
 q.fill_(float('nan'));k.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
 assert torch.count_nonzero(q)==torch.count_nonzero(k)==0
 del graph,x,q,k
 from torch._inductor.runtime.triton_heuristics import CachingAutotuner
 original=CachingAutotuner.run;classified={};shapes=set();local=threading.local()
 signature=['in_ptr0','out_ptr0','out_ptr1','xnumel_0','xnumel_1','XBLOCK','R0_BLOCK']
 _state=dict(source=str(p),source_sha256=SOURCE_SHA,config=CONFIG,combined_function_ast_sha256=COMBINED_AST,
             graph_replay_poison_check=True,kernels=[],calls_by_rows={},captured_rows=[],original_combined_launches_replaced=0,separate_pairs_replaced=0,pending_pairs=0)
 def run(kernel,*inputs,stream,benchmark_run=False,**kwargs):
  if kernel not in classified:
   name=kernel.fn.__name__
   kind={'triton_red_fused_6':'combined',
         'triton_red_fused_clone_rms_norm_split_split_with_sizes_view_6':'q',
         'triton_red_fused_rms_norm_split_with_sizes_view_7':'k'}.get(name)
   if kind:
    expected=signature if kind=='combined' else ['in_ptr0','out_ptr0','xnumel','r0_numel','XBLOCK','R0_BLOCK']
    assert list(kernel.triton_meta['signature'])==expected
    fn=next(n for n in ast.parse(kernel.fn.src).body if isinstance(n,ast.FunctionDef));fn.decorator_list=[]
    digest=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
    assert digest=={'combined':COMBINED_AST,'q':Q_AST,'k':K_AST}[kind],digest
    _state['kernels'].append(dict(kind=kind,function=name,filename=kernel.filename,source_ast_sha256=digest))
   classified[kernel]=kind
  kind=classified[kernel];pending=getattr(local,'pending',None)
  if pending is not None:assert kind=='k','Q/K pair interrupted by another Triton call'
  if kind is None:return original(kernel,*inputs,stream=stream,benchmark_run=benchmark_run,**kwargs)
  assert not kwargs and not benchmark_run
  captured=torch.cuda.is_current_stream_capturing()
  if kind=='q':
   assert pending is None and len(inputs)==4 and inputs[3]==256 and inputs[2]%16==0
   local.pending=(inputs,stream,captured);_state['pending_pairs']+=1
   return None
  if kind=='k':
   assert pending is not None and len(inputs)==4 and inputs[3]==256
   qargs,qstream,qcaptured=pending
   assert qargs[0] is inputs[0] and qargs[2]==8*inputs[2]
   assert qstream==stream and qcaptured==captured
   merged=(inputs[0],qargs[1],inputs[1],qargs[2],inputs[2])
   launch(merged,stream);local.pending=None;_state['pending_pairs']-=1
   _state['separate_pairs_replaced']+=1;T=int(inputs[2])//2
  else:
   launch(inputs,stream);_state['original_combined_launches_replaced']+=1;T=int(inputs[4])//2
  key=str(T);_state['calls_by_rows'][key]=_state['calls_by_rows'].get(key,0)+1
  if captured and T not in _state['captured_rows']:_state['captured_rows'].append(T)
  if (T,captured) not in shapes:shapes.add((T,captured));persist()
  return None
 CachingAutotuner.run=run;_installed=True;persist()
