"""Candidate: materialize the initial prompt boundary that sparse retention keeps.

The global speculative-decoding flag is unchanged. Only the read-only view sent
to the existing model-scoped Mamba alignment helper can differ.
"""
import hashlib,inspect
from pathlib import Path

PREFIX_SHA='65fb972e285e37741f53b221d2f6380ec464d34a39f36bb66a82dcaacab727d0'
SCHEDULER_SHA='30160d4df0366e22f9ff583d991e2898b3b65c5d25e471a209578336e24001b5'

def should_remove_backoff(self,request):
    config=self.vllm_config
    spec=config.speculative_config
    return (getattr(config.model_config,'quantization',None)=='ornith_g256'
        and config.cache_config.enable_prefix_caching
        and config.cache_config.mamba_cache_mode=='align'
        and not config.additional_config.get('ornith_g256',{}).get('draft_full_retention',False)
        and self.block_size==1120 and not self.mamba_partial_cache_hit
        and spec is not None and spec.method=='dflash' and spec.num_speculative_tokens==15
        and self.use_eagle and not self.mamba_has_prefill_checkpoint_blocks
        and request.num_tokens==request.num_prompt_tokens
        and request.num_computed_tokens<request.num_prompt_tokens
        and request.num_tokens%self.block_size!=0)

def install():
    from ornith_g256 import prefix_cache
    from vllm.v1.core.sched.scheduler import Scheduler
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(prefix_cache.__file__)==PREFIX_SHA
    prefix_cache._install_mamba_alignment()
    original=prefix_cache._upstream_mamba_split
    assert sha(inspect.getsourcefile(original))==SCHEDULER_SHA
    assert Scheduler._mamba_block_aligned_split is prefix_cache._mamba_block_aligned_split
    counts=dict(calls=0,scoped_calls=0)
    def split(self,request,num_new_tokens,num_new_local_computed_tokens=0,num_external_computed_tokens=0):
        counts['calls']+=1
        if not should_remove_backoff(self,request):
            return original(self,request,num_new_tokens,num_new_local_computed_tokens,num_external_computed_tokens)
        counts['scoped_calls']+=1
        view=prefix_cache._AttributeView(self,use_eagle=False)
        result=original(view,request,num_new_tokens,num_new_local_computed_tokens,num_external_computed_tokens)
        return result
    prefix_cache._upstream_mamba_split=split
    return counts
