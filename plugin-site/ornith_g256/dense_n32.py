"""Isolated M1 dense N32 shadows; retain N16 verification and prefill."""
# Copyright 2026 Ciru.
import ctypes as C
import json
from pathlib import Path
import torch
_INSTALLED=False
_BANKS={}
_TOTAL=0
_LIB=None
_LAUNCH=None

def install():
 global _INSTALLED,_LIB,_LAUNCH
 if _INSTALLED:return
 from . import native
 from .method import G256LinearMethod
 library=Path(__file__).resolve().parents[2]/'native/libornith_dense_g256_n32.so'
 _LIB=C.CDLL(str(library.resolve(strict=True)))
 _LAUNCH=_LIB.ornith_dense_g256_launch_n32
 _LAUNCH.argtypes=[C.c_void_p]*4+[C.c_size_t]+[C.c_void_p]*2+[C.c_int]*7+[C.c_void_p]
 _LAUNCH.restype=C.c_int
 original_load=G256LinearMethod.process_weights_after_loading
 def load(self,layer):
  global _TOTAL
  original_load(self,layer)
  if self.fields != ('g256_codes','g256_metadata'):return
  n,k=self.n,self.k
  if n%32 or not 0<n<=12288 or k%256 or not 0<k<=8192:
   raise ValueError(f'Dense N32 unsupported N{n}/K{k}')
  codes=layer.g256_codes;meta=layer.g256_metadata
  c32=codes.reshape(n//32,2,k//256,32,16).permute(0,2,3,1,4).contiguous().reshape(n//32,k//256,32,32)
  m32=meta.reshape(n//32,2,k//256,16).permute(0,2,1,3).contiguous().reshape(n//32,k//256,32)
  layer.register_buffer('_dense_n32_codes',c32,persistent=False)
  layer.register_buffer('_dense_n32_metadata',m32,persistent=False)
  key=codes.data_ptr()
  if key in _BANKS:raise RuntimeError('Dense N32 bank prepared twice')
  _BANKS[key]=(c32,m32)
  _TOTAL+=c32.numel()*c32.element_size()+m32.numel()*m32.element_size()
  if len(_BANKS) in (1,160):
   print('ORNITH_DENSE_N32 '+json.dumps({'matrices':len(_BANKS),'extra_payload_bytes':_TOTAL,'last_shape':[n,k],'scope':'M1 only; N16 retained'}),flush=True)
 G256LinearMethod.process_weights_after_loading=load
 original_dense=native._dense_impl
 def dense(x,codes,metadata,workspace,out,flags,capacity,n,k,geometry,a8_max_rows):
  if x.shape[0]==1 and a8_max_rows>=1 and geometry==2:
   bank=_BANKS.get(codes.data_ptr())
   if bank is None:raise RuntimeError('Missing dense N32 shadow for M1')
   native.validate(x,out,capacity,k)
   native.check(_LAUNCH(native.ptr(x),native.ptr(bank[0]),native.ptr(bank[1]),native.ptr(workspace),workspace.numel(),native.ptr(out),native.ptr(flags),1,capacity,n,k,8,128,2,native.stream(x)),'G256 dense N32 M1')
   return
  return original_dense(x,codes,metadata,workspace,out,flags,capacity,n,k,geometry,a8_max_rows)
 native._dense_impl=dense
 _INSTALLED=True
