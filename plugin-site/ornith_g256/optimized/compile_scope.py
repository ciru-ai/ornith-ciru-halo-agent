"""Keep model-only fusion hooks outside compiler autotuning microbenchmarks."""
import json
import threading
from pathlib import Path

def install(original_run, output):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    selected_run=CachingAutotuner.run
    original_tuning=PythonWrapperCodegen.generate_and_run_autotune_block
    local=threading.local()
    state=dict(autotune_blocks=0,original_autotune_calls=0,active=False)
    out=Path(output)/'compile-scope.json'
    def persist():out.write_text(json.dumps(state,indent=2)+'\n')
    def tuning(wrapper,*args,**kwargs):
        previous=getattr(local,'autotuning',False)
        local.autotuning=True;state['active']=True
        state['autotune_blocks']+=1
        try:return original_tuning(wrapper,*args,**kwargs)
        finally:
            local.autotuning=previous;state['active']=previous;persist()
    def run(kernel,*args,**kwargs):
        if getattr(local,'autotuning',False):
            state['original_autotune_calls']+=1
            return original_run(kernel,*args,**kwargs)
        return selected_run(kernel,*args,**kwargs)
    PythonWrapperCodegen.generate_and_run_autotune_block=tuning
    CachingAutotuner.run=run
    persist()
    return lambda:dict(state)
