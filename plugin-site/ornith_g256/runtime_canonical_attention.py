"""Preserve released single-query attention partitions in speculative decode."""
def install_worker_hook(worker):
    original=worker.init_device
    def initialize(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        from . import attention_folded,attention_partition
        attention_folded.forward=attention_partition.forward
        print('ORNITH_CANONICAL_RELEASED_ATTENTION',flush=True)
        return result
    worker.init_device=initialize
