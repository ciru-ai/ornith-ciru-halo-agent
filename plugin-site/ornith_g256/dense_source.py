"""Original BF16 selected dense weights for prefill; verification stays packed."""
from pathlib import Path
import torch
from safetensors import safe_open
_INSTALLED=False
_WEIGHTS={}
def install():
 global _INSTALLED
 if _INSTALLED:return
 from . import native
 from .method import G256LinearMethod
 path=Path(__file__).resolve().parents[2]/'models/source-prefill.safetensors'
 original_load=G256LinearMethod.process_weights_after_loading
 def load(self,layer):
  original_load(self,layer)
  if not self.prefix.endswith(('.linear_attn.in_proj_qkvz','.mlp.shared_expert.gate_up_proj','.mlp.shared_expert.down_proj')):return
  name=self.prefix.replace('language_model.model.','model.language_model.',1)
  with safe_open(str(path),framework='pt',device='cpu') as f:w=f.get_tensor(name)
  assert w.dtype==torch.bfloat16 and tuple(w.shape)==(self.n,self.k)
  w=w.to(device=layer.g256_codes.device)
  layer.register_buffer('_source_prefill_weight',w,persistent=False)
  _WEIGHTS[layer.g256_codes.data_ptr()]=w
  if len(_WEIGHTS) in (1,110):print('ORNITH_SOURCE_PREFILL',len(_WEIGHTS),sum(t.numel()*t.element_size() for t in _WEIGHTS.values()),flush=True)
 G256LinearMethod.process_weights_after_loading=load
 original_dense=native._dense_impl
 def dense(x,codes,metadata,workspace,out,flags,capacity,n,k,geometry,a8_max_rows):
  # M64 complete-operation measurements favor these two source shapes.
  # The resident weights already exist for prefill; no additional allocation.
  if x.shape[0]>a8_max_rows or (x.shape[0]==64 and
      (n,k) in ((12288,2048),(1024,2048))):
   weight=_WEIGHTS.get(codes.data_ptr())
   if weight is not None:
    torch.mm(x,weight.t(),out=out)
    return
  return original_dense(x,codes,metadata,workspace,out,flags,capacity,n,k,geometry,a8_max_rows)
 native._dense_impl=dense
 _INSTALLED=True
