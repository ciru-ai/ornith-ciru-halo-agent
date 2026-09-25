"""Isolated GDN spatial traversal adapter; leaves postconv/h/cumsum untouched.

Install with enabled=False during normal model warmup. Candidate dispatch reuses
only configurations already selected by the original Autotuner. A missing key
fails before compilation/benchmarking of alternative configurations.
"""
from __future__ import annotations
import copy
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import json

class ParentConfigurationRequired(Exception):
    pass

ROOT = Path(__file__).resolve().parent
OPS = 'vllm.third_party.flash_linear_attention.ops'
SPECS = {
    'chunk_scaled_dot_kkt_fwd': (OPS+'.chunk_scaled_dot_kkt','chunk_scaled_dot_kkt_fwd_kernel'),
    'solve_tril': (OPS+'.solve_tril','merge_16x16_to_64x64_inverse_kernel'),
    'recompute_w_u_fwd': (OPS+'.wy_fast','recompute_w_u_fwd_kernel'),
    'chunk_fwd_o': (OPS+'.chunk_o','chunk_fwd_kernel_o'),
    'causal_conv1d_fn': ('vllm.model_executor.layers.mamba.ops.causal_conv1d','_causal_conv1d_fwd_kernel'),
}

def exact_tuner_key(tuner, args, kwargs):
    # Exact installed Autotuner.run key construction, including dtype order.
    nargs = dict(zip(tuner.arg_names,args))
    all_args = {**nargs,**kwargs}
    selected = {k:v for k,v in all_args.items() if k in tuner.arg_names}
    key = [selected[k] for k in tuner.keys if k in selected]
    for value in selected.values():
        if hasattr(value,'dtype'): key.append(str(value.dtype))
    return tuple(key)

def safe_conv_scope(arguments):
    # Only one live sequence, no APC intermediate cache writes and no x/state alias.
    if any(arguments.get(n) is not None for n in ('block_idx_first_scheduled_token','block_idx_last_scheduled_token','initial_state_idx','num_computed_tokens')):
        return False
    x = arguments['x']; starts=arguments['query_start_loc']; state=arguments['conv_states']
    if tuple(starts.shape)!=(2,) or x.ndim!=2 or x.shape[1]<=8:
        return False
    if arguments.get('has_initial_state') is not None and tuple(arguments['has_initial_state'].shape)!=(1,):
        return False
    if x.untyped_storage().data_ptr()==state.untyped_storage().data_ptr():
        return False
    return True

class SpatialPipeline:
    def __init__(self, enabled):
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
        self.enabled=enabled
        self.patches=[]
        self.original={}
        self.candidate={}
        self.tuners={}
        self.calls={name:{'parent':0,'candidate':0,'scope_fallback':0} for name in SPECS}
        self.activations=[]
        self.configs=[]
        self.seen=set()
        proof=json.loads((ROOT/'SOURCE-PROOF.json').read_text())
        for stage,(module_name,kernel_name) in SPECS.items():
            parent=importlib.import_module(module_name)
            filename=module_name.rsplit('.',1)[1]+'.py'
            assert hashlib.sha256(Path(parent.__file__).read_bytes()).hexdigest()==proof[filename]['parent_sha256'],('Installed source changed',module_name)
            path=ROOT/'candidate'/filename
            assert hashlib.sha256(path.read_bytes()).hexdigest()==proof[filename]['candidate_sha256']
            # Loading inside the real package preserves every unchanged relative import.
            name=module_name.rsplit('.',1)[0]+'._gdn_spatial_'+filename[:-3]
            spec=importlib.util.spec_from_file_location(name,path)
            candidate=importlib.util.module_from_spec(spec)
            sys.modules[name]=candidate
            spec.loader.exec_module(candidate)
            new_leaf=getattr(candidate,kernel_name)
            while not isinstance(new_leaf,JITFunction): new_leaf=new_leaf.fn
            old_kernel=getattr(parent,kernel_name)
            def clone(node, *, stage=stage, new_leaf=new_leaf):
                if isinstance(node,JITFunction):
                    assert node.arg_names==new_leaf.arg_names
                    return new_leaf
                duplicate=copy.copy(node)
                duplicate.fn=clone(node.fn)
                if isinstance(node,Autotuner):
                    assert duplicate.cache is node.cache and duplicate.configs is node.configs
                    duplicate.base_fn=new_leaf.fn
                    self.tuners[stage]=(node,duplicate)
                    run_impl=type(node).run
                    def guarded_run(this,*args,**kwargs):
                        key=exact_tuner_key(this,args,kwargs)
                        if len(this.configs)>1 and key not in this.cache:
                            raise ParentConfigurationRequired(stage, key)
                        config=this.cache[key] if len(this.configs)>1 else this.configs[0]
                        record_key=(stage,repr(key),repr(config.all_kwargs()))
                        if record_key not in self.seen:
                            self.seen.add(record_key)
                            self.configs.append(dict(stage=stage,key=repr(key),parent_selected_config=config.all_kwargs(),same_config_object=True))
                        return run_impl(this,*args,**kwargs)
                    duplicate.run=types.MethodType(guarded_run,duplicate)
                return duplicate
            candidate.__dict__[kernel_name]=clone(old_kernel)
            self.original[stage]=getattr(parent,stage)
            self.candidate[stage]=getattr(candidate,stage)
        from vllm.third_party.flash_linear_attention.ops import chunk
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
        chunk_globals=chunk.chunk_gated_delta_rule_fwd.__globals__
        core_globals=qwen.QwenGatedDeltaNetAttention._forward_core.__globals__
        # Validate the whole hook set before the first global mutation.
        for stage in SPECS:
            namespace=core_globals if stage=='causal_conv1d_fn' else chunk_globals
            if namespace.get(stage) is not self.original[stage]:
                raise RuntimeError(f'Unexpected actual installed caller alias for {stage}')
        for stage in SPECS:
            namespace=core_globals if stage=='causal_conv1d_fn' else chunk_globals
            original=self.original[stage]
            signature=inspect.signature(original)
            def dispatch(*args,_stage=stage,_signature=signature,**kwargs):
                active=bool(self.enabled())
                if active:
                    bound=_signature.bind(*args,**kwargs);bound.apply_defaults()
                    values=bound.arguments
                    if _stage=='causal_conv1d_fn':
                        safe=safe_conv_scope(values)
                        rows=int(values['x'].shape[1])
                    elif _stage=='solve_tril':
                        rows=int(values['A'].shape[1])
                        safe=values['A'].shape[-1]==64 and rows>8
                    else:
                        input_tensor=values['q'] if _stage=='chunk_fwd_o' else values['k']
                        rows=int(input_tensor.shape[1]);safe=rows>8
                    if not safe:
                        self.calls[_stage]['scope_fallback']+=1
                        active=False
                    else:
                        key=('activation',_stage,rows)
                        if key not in self.seen:
                            self.seen.add(key)
                            self.activations.append(dict(stage=_stage,rows=rows,scope='single-sequence non-APC' if _stage=='causal_conv1d_fn' else 'independent chunk/head tiles'))
                arm='candidate' if active else 'parent'
                self.calls[_stage][arm]+=1
                if not active:
                    return self.original[_stage](*args, **kwargs)
                try:
                    return self.candidate[_stage](*args, **kwargs)
                except ParentConfigurationRequired:
                    self.calls[_stage]['candidate'] -= 1
                    self.calls[_stage]['parent'] += 1
                    self.calls[_stage]['tuner_cold'] = self.calls[_stage].get('tuner_cold', 0) + 1
                    return self.original[_stage](*args, **kwargs)
            dispatch.__name__=stage
            dispatch.__signature__=signature
            self.patches.append((namespace,stage,original))
            namespace[stage]=dispatch

    def snapshot(self):
        return dict(calls=copy.deepcopy(self.calls),activations=copy.deepcopy(self.activations),
                    parent_selected_configs=copy.deepcopy(self.configs),
                    shared_autotuner_cache={name:parent.cache is candidate.cache for name,(parent,candidate) in self.tuners.items()},
                    unchanged=['postconv','chunk_local_cumsum','chunk_gated_delta_rule_fwd_h','convolution update/decode','solve BT16/BT32'])

    def restore(self):
        for namespace,name,original in reversed(self.patches): namespace[name]=original
        self.patches.clear()

def install(enabled):
    """Return a handle exposing snapshot()/restore(); enabled is a no-argument callable."""
    return SpatialPipeline(enabled)
