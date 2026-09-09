"""C8 recurrent update logs; reconstruct before upstream cache alignment."""
# Copyright 2026 Ciru.
from functools import wraps
import inspect
import torch
from . import gdn_compact_kernel as kernel
_ACTIVE=False
_CAPTURE=False
_INSTALLED=False
_BANKS={}
_LAYER=None

def install():
    global _INSTALLED
    if _INSTALLED:return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker import mamba_utils
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    cls=qwen_gdn_linear_attn.QwenGatedDeltaNetAttention
    core_original=cls._forward_core
    @wraps(core_original)
    def core(self,*args,**kwargs):
        global _LAYER
        previous=_LAYER;_LAYER=self.prefix
        try:return core_original(self,*args,**kwargs)
        finally:_LAYER=previous
    cls._forward_core=core
    original=qwen_gdn_linear_attn.fused_sigmoid_gating_delta_rule_update
    signature=inspect.signature(original)
    def update(*args,**kwargs):
        if not _ACTIVE:return original(*args,**kwargs)
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();d=dict(bound.arguments)
        state=d['initial_state'];q=d['q'];v=d['v'];cu=d['cu_seqlens'];ids=d['ssm_state_indices']
        if cu is None or ids is None or ids.ndim!=2 or d['is_kda']:
            raise RuntimeError('Compact GDN requires varlen indexed scalar decay')
        N=len(cu)-1;HV=v.shape[2];V=v.shape[-1];K=q.shape[-1]
        assert N<=8 and q.shape[1]<=128 and ids.shape[1]<=16 and (HV,V,K)==(32,128,128)
        key=_LAYER
        assert key is not None
        if key not in _BANKS:
            print('ORNITH_GDN_BANK',key,state.data_ptr(),tuple(state.shape),tuple(state.stride()),flush=True)
            make=lambda *shape:torch.empty(*shape,device=q.device,dtype=torch.float32)
            _BANKS[key]=dict(base=make(8,HV,V,K),keys=make(128,HV,K),values=make(128,HV,V),decays=make(128,HV),
                             cu=torch.empty(9,device=q.device,dtype=cu.dtype),
                             ids=torch.empty(8,16,device=q.device,dtype=ids.dtype),state=state)
        bank=_BANKS[key]
        bank['cu'][:N+1].copy_(cu)
        bank['ids'][:N,:ids.shape[1]].copy_(ids)
        return kernel.fused_sigmoid_gating_delta_rule_update(**d,compact_base=bank['base'],
            compact_k=bank['keys'],compact_v=bank['values'],compact_g=bank['decays'])
    qwen_gdn_linear_attn.fused_sigmoid_gating_delta_rule_update=update
    execute_original=GPUModelRunner.execute_model
    capture_original=GPUModelRunner._warmup_and_capture
    dispatch_original=GPUModelRunner._determine_batch_execution_and_padding
    post_original=mamba_utils.postprocess_mamba_align_gpu
    @wraps(execute_original)
    def execute(self,scheduler_output,*args,**kwargs):
        global _ACTIVE
        counts=scheduler_output.num_scheduled_tokens;cached=scheduler_output.scheduled_cached_reqs
        _ACTIVE=(bool(counts) and not scheduler_output.scheduled_new_reqs and
                 all(c==8 for c in counts.values()) and
                 all(r in cached.req_ids and not cached.is_context_phase(r) for r in counts))
        return execute_original(self,scheduler_output,*args,**kwargs)
    @wraps(capture_original)
    def capture(self,desc,*args,**kwargs):
        global _ACTIVE,_CAPTURE
        previous=(_ACTIVE,_CAPTURE)
        _CAPTURE=True
        _ACTIVE=bool(desc.uniform and desc.num_reqs and desc.num_tokens==desc.num_reqs*8)
        try:return capture_original(self,desc,*args,**kwargs)
        finally:_ACTIVE,_CAPTURE=previous
    @wraps(dispatch_original)
    def dispatch(self,num_tokens,num_reqs,num_scheduled_tokens_np,max_num_scheduled_tokens,*args,**kwargs):
        # A prompt whose shape coincides with Q8 must not replay a compact decode graph.
        if not _CAPTURE and not _ACTIVE and max_num_scheduled_tokens==8:
            kwargs['force_eager']=True
        return dispatch_original(self,num_tokens,num_reqs,num_scheduled_tokens_np,max_num_scheduled_tokens,*args,**kwargs)
    @wraps(post_original)
    def post(**kwargs):
        global _ACTIVE
        if _ACTIVE:
            assert len(_BANKS)==30, f'Expected30targetGDN layers, got{len(_BANKS)}'
            ctx=kwargs['bufs'].postprocess_align
            for bank in _BANKS.values():
                state=bank['state']
                kernel.replay_physical[(4,kwargs['num_reqs']*32)](
                    bank['base'],bank['keys'],bank['values'],bank['decays'],
                    kwargs['num_accepted_tokens_gpu'],bank['cu'],bank['ids'],state,
                    ctx.num_computed_tokens_buf.gpu,ctx.num_scheduled_tokens_buf.gpu,ctx.num_draft_tokens_buf.gpu,
                    32,128,128,state.stride(0),1120,32,num_warps=4,num_stages=3)
            if not getattr(post,'logged',False):
                print('ORNITH_COMPACT_GDN replayed30layers before cache alignment',flush=True);post.logged=True
        try:return post_original(**kwargs)
        finally:_ACTIVE=False
    GPUModelRunner.execute_model=execute
    GPUModelRunner._warmup_and_capture=capture
    GPUModelRunner._determine_batch_execution_and_padding=dispatch
    mamba_utils.postprocess_mamba_align_gpu=post
    _INSTALLED=True
