"""Keep verification precision stable when new prompts join a target batch."""
from functools import wraps
_installed=False

def install():
    global _installed
    if _installed:return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from . import native
    prepare_original=GPUModelRunner._prepare_inputs
    execute_original=GPUModelRunner.execute_model
    @wraps(prepare_original)
    def prepare(self,scheduler_output,num_scheduled_tokens):
        b=self.input_batch;n=b.num_reqs
        phases=b.num_computed_tokens_cpu[:n] < b.num_prompt_tokens[:n]
        counts=[int(x) for x in num_scheduled_tokens]
        segments=[];requests=[];offset=0
        for req,(count,prefill) in enumerate(zip(counts,phases)):
            phase=bool(prefill)
            requests.append((offset,offset+count,phase,
                             int(b.num_computed_tokens_cpu[req])+count))
            if segments and segments[-1][2]==phase:
                segments[-1]=(segments[-1][0],offset+count,phase)
            else:segments.append((offset,offset+count,phase))
            offset+=count
        native._PHASE_SEGMENTS=tuple(segments) if offset>64 and any(phases) and not all(phases) else ()
        # Same CPU metadata used by vLLM optimistic_seq_lens_cpu. Actual GPU
        # seq_lens remain authoritative for attention; only prefill uses the
        # exact CPU length for existing fixed-width IU4 eligibility.
        native._ATTENTION_PHASE_REQUESTS=(tuple(requests)
            if any(phases) and not all(phases) and not self.use_async_scheduling else ())
        if native._PHASE_SEGMENTS:
            print('ORNITH_PHASE_DISPATCH '+str(native._PHASE_SEGMENTS),flush=True)
        return prepare_original(self,scheduler_output,num_scheduled_tokens)
    @wraps(execute_original)
    def execute(self,*args,**kwargs):
        native._PHASE_SEGMENTS=()
        native._ATTENTION_PHASE_REQUESTS=()
        try:return execute_original(self,*args,**kwargs)
        finally:
            native._PHASE_SEGMENTS=()
            native._ATTENTION_PHASE_REQUESTS=()
    GPUModelRunner._prepare_inputs=prepare
    GPUModelRunner.execute_model=execute
    _installed=True
