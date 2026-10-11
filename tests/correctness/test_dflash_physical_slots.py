"""CPU integration checks. Fake tensors/launcher do not qualify a GPU kernel."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'tests/correctness/mm-slots-v416-parent'
PLUGIN = ROOT/'plugin-site/ornith_g256'
ADAPTER = PLUGIN/'iu4_correctness/dflash_physical_slots.py'
NATIVE = ROOT/'tests/correctness/mm-slots-v416-native'


class Tensor:
    def __init__(self, shape, values=None, dtype='i64', pointer=None, strides=None, device='cuda'):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = NS(type=device)
        self.values = values
        self.pointer = id(self) if pointer is None else pointer
        self.strides = strides or ((1,) if len(shape)==1 else (shape[1],1))
    def numel(self):
        result=1
        for n in self.shape: result*=n
        return result
    def stride(self, dim): return self.strides[dim]
    def data_ptr(self): return self.pointer
    def __getitem__(self, part):
        values=self.values[part] if self.values is not None else None
        n=len(values) if values is not None else part.stop
        return Tensor((n,),values,self.dtype,self.pointer)
    def item(self): raise AssertionError('GPU scalar read forbidden')
    def cpu(self): raise AssertionError('GPU readback forbidden')
    def tolist(self): raise AssertionError('GPU readback forbidden')


TORCH = ModuleType('torch')
TORCH.Tensor = Tensor
TORCH.int32 = 'i32'
TORCH.int64 = 'i64'
TORCH.empty = lambda shape, dtype, device: Tensor(shape, dtype=dtype, device=device.type)
TRITON = NS(jit=lambda fn:fn, cdiv=lambda a,b:(a+b-1)//b)
VLLM_TRITON = ModuleType('vllm.triton_utils')
VLLM_TRITON.tl = NS(constexpr=int)
VLLM_TRITON.triton = TRITON


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch':TORCH,'vllm.triton_utils':VLLM_TRITON}):
        spec.loader.exec_module(module)
    return module


class ScalarTL:
    where=staticmethod(lambda condition,a,b:a if condition else b)


def scalar_expressions():
    fn=next(n for n in ast.parse(ADAPTER.read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name=='_physical_slots')
    values={n.targets[0].id:n.value for n in ast.walk(fn)
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert ast.unparse(values['position'])=='tl.where(context, seq - count + j, seq - rejected + query_index)'
    assert ast.unparse(values['slot'])=='tl.where(valid & (page > 0), page * PAGE + position % PAGE, -1)'
    return {k:compile(ast.Expression(values[k]),'<actual-adapter-kernel-expression>','eval')
            for k in ('position','valid','slot')}


class Launcher:
    def __init__(self): self.calls=[];self.expr=scalar_expressions()
    def __getitem__(self,grid):
        def launch(starts,seq,table,rejected,contexts,queries,stride,cols,page_size,width,
                   context_size,has_rejected,block,**kwargs):
            self.calls.append(dict(grid=grid,width=width,has_rejected=has_rejected,
                                   context_size=context_size,kwargs=kwargs))
            for req in range(grid[0]):
                begin,end=starts.values[req:req+2]
                count=end-begin
                rejected_count=rejected.values[req] if has_rejected else 0
                for j in range(grid[1]*block):
                    context=j<count;query=count<=j<count+width
                    env=dict(tl=ScalarTL,context=context,query=query,count=count,seq=seq.values[req],
                             j=j,rejected=rejected_count,query_index=j-count,COLS=cols,PAGE=page_size)
                    env['position']=eval(self.expr['position'],{},env)
                    env['valid']=eval(self.expr['valid'],{},env)
                    env['page']=table.values[req][env['position']//page_size] if env['valid'] else 0
                    slot=eval(self.expr['slot'],{},env)
                    if context and 0<=begin+j<context_size: contexts.values[begin+j]=slot
                    if query: queries.values[req*width+j-count]=slot
        return launch


def geometry(requests=1,width=4,seq_length=9000,count=4,rejected_count=0,gap=240):
    starts=Tensor((requests+1,),[i*count for i in range(requests+1)],'i32')
    seq=Tensor((requests,),[seq_length]*requests,'i32')
    cols=(seq_length+width+559)//560
    table=Tensor((requests,cols),[[100+r*1000+7*j for j in range(cols)] for r in range(requests)],'i32')
    context=Tensor((8192,),[-999]*8192)
    query=Tensor((128,),[-999]*128)
    draft=NS(num_speculative_tokens=width-1,block_size=560,kv_cache_gid=2,
             draft_attn_groups=[NS(kv_cache_group_id=2)],
             _context_slot_mapping_buffer=context,_slot_mapping_buffer=query,
             _per_group_slot_mappings={2:query})
    view=Tensor((requests*width,),pointer=query.pointer)
    result=(requests*width,object(),NS(slot_mapping=view))
    cad=NS(batch_size=lambda:requests,max_query_len=count,query_start_loc=starts,
           seq_lens=seq,block_table_tensor=table)
    tokens=Tensor((requests*count,),list(range(requests*count)))
    positions=Tensor((3,requests*count),[[seq_length-count-gap+j for j in range(count)]*requests]*3)
    rejected=Tensor((requests,),[rejected_count]*requests,'i32') if rejected_count else None
    return draft,cad,tokens,positions,rejected,result


class PhysicalSlotsTests(unittest.TestCase):
    def setUp(self): self.module=load(ADAPTER,'cpu_physical_slots');self.launcher=Launcher();self.module._physical_slots=self.launcher

    def test_native_and_hook_source_pins_are_actual_parent(self):
        for name,filename in [('dflash','sozo-installed-dflash-as-read.py'),('utils','sozo-installed-spec-utils-as-read.py')]:
            self.assertEqual(hashlib.sha256((NATIVE/filename).read_bytes()).hexdigest(),self.module.NATIVE_SOURCE_SHA256[name])
        for name,expected in self.module.HOOK_SOURCE_SHA256.items():
            self.assertEqual(hashlib.sha256((PLUGIN/(name+'.py')).read_bytes()).hexdigest(),expected)

    def test_candidate_source_pins_exact_delta(self):
        old=json.loads((PARENT/'iu4_correctness/source-pins.json').read_text())['files']
        new=json.loads((PLUGIN/'iu4_correctness/source-pins.json').read_text())['files']
        self.assertEqual((len(old),len(new)),(103,104))
        self.assertEqual(set(new)-set(old),{'iu4_correctness/dflash_physical_slots.py'})
        self.assertEqual({name for name in old if old[name]!=new[name]},{'iu4_correctness/worker.py'})
        self.assertEqual(sum(old[name]==new[name] for name in old),102)
        for root,pins in ((PARENT,old),(PLUGIN,new)):
            for name,digest in pins.items():
                path = root/name
                if root == PARENT and not path.exists(): path = PLUGIN/name
                if name == '_xgrammar_native/libxgrammar-candidate.so' and not path.exists():
                    self.assertEqual(digest,'92ab36bfae20c6c4df3f5321ca375204799ac833252921d9d8f6a206491d35f7')
                    continue
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest,name)

    def invoke(self,rows):
        draft,cad,tokens,positions,rejected,result=rows
        trace=[]
        def previous(*args): trace.append(args);return result
        wrapper=MethodType(self.module._wrap_inputs(draft,previous),draft)
        objects=[tokens,object(),positions,object(),object(),cad,rejected]
        actual=wrapper(*objects)
        self.assertIs(actual,result)
        self.assertEqual(len(trace),1)
        self.assertTrue(all(a is b for a,b in zip(objects,trace[0])))
        return draft,cad

    def test_mm_context_query_slots_use_physical_lengths_not_rope(self):
        rows=geometry();positions=rows[3];before=list(positions.values[0])
        draft,cad=self.invoke(rows)
        seq=cad.seq_lens.values[0];table=cad.block_table_tensor.values[0]
        expected=[table[p//560]*560+p%560 for p in range(seq-4,seq+4)]
        self.assertEqual(draft._context_slot_mapping_buffer.values[:4],expected[:4])
        self.assertEqual(draft._slot_mapping_buffer.values[:4],expected[4:])
        self.assertEqual(positions.values[0],before)
        self.assertEqual(self.launcher.calls[0]['grid'],(1,1))

    def test_text_page_crossing_is_negative_control(self):
        draft,cad=self.invoke(geometry(seq_length=562,gap=0))
        table=cad.block_table_tensor.values[0]
        expected=[table[p//560]*560+p%560 for p in range(558,566)]
        self.assertEqual(draft._context_slot_mapping_buffer.values[:4],expected[:4])
        self.assertEqual(draft._slot_mapping_buffer.values[:4],expected[4:])

    def test_rejected_tail_changes_query_start_only(self):
        draft,cad=self.invoke(geometry(requests=2,width=8,count=8,rejected_count=3))
        for r in range(2):
            table=cad.block_table_tensor.values[r]
            self.assertEqual(draft._slot_mapping_buffer.values[r*8:(r+1)*8],
                             [table[p//560]*560+p%560 for p in range(8997,9005)])
        self.assertTrue(self.launcher.calls[0]['has_rejected'])

    def test_empty_padded_rows_cannot_reuse_previous_query_slots(self):
        rows=geometry(requests=2)
        rows[1].query_start_loc.values=[0,4,4]
        rows[1].seq_lens.values[1]=0
        self.invoke(rows)
        self.assertEqual(rows[0]._slot_mapping_buffer.values[4:8],[-1]*4)

    def test_out_of_range_positions_write_padding_not_clamped_alias(self):
        rows=geometry(seq_length=560)
        rows[1].block_table_tensor=Tensor((1,1),[[117]],'i32')
        self.invoke(rows)
        self.assertEqual(rows[0]._slot_mapping_buffer.values[:4],[-1]*4)

    def test_retired_null_pages_are_masked_with_absolute_table_indices(self):
        rows=geometry(seq_length=65535,count=8192)
        table=rows[1].block_table_tensor
        table.values[0][:110]=[0]*110
        table.strides=(table.shape[1]+3,1)
        draft,cad=self.invoke(rows)
        begin=65535-8192
        expected=[-1 if table.values[0][p//560]<=0 else table.values[0][p//560]*560+p%560
                  for p in range(begin,65535)]
        self.assertEqual(draft._context_slot_mapping_buffer.values,expected)
        self.assertTrue(any(slot==-1 for slot in expected))
        self.assertNotEqual(expected[-1],-1)
        self.assertEqual(self.launcher.calls[0]['grid'],(1,33))
        table.values[0][65535//560]=0
        self.invoke(rows)
        self.assertEqual(draft._slot_mapping_buffer.values[:4],[-1]*4)

    def test_eight_mixed_rows_q16_rejections_and_padding(self):
        rows=list(geometry(requests=8,width=16,seq_length=131040,count=16,rejected_count=1))
        counts=[0,1,9,4,16,7,2,8];starts=[0]
        for count in counts:starts.append(starts[-1]+count)
        rows[1].query_start_loc.values=starts
        rows[1].seq_lens.values=[0,560,8193,9000,65535,87335,128001,131040]
        rows[4].values=[0,0,3,1,7,2,0,4]
        rows[2]=Tensor((starts[-1],),list(range(starts[-1])))
        draft,cad=self.invoke(rows)
        for r,count in enumerate(counts):
            seq=cad.seq_lens.values[r];rejected=rows[4].values[r];table=cad.block_table_tensor.values[r]
            expected=[table[p//560]*560+p%560 for p in range(seq-rejected,seq-rejected+16)] if count else [-1]*16
            self.assertEqual(draft._slot_mapping_buffer.values[r*16:(r+1)*16],expected)
            expected_context=[table[p//560]*560+p%560 for p in range(seq-count,seq)]
            self.assertEqual(draft._context_slot_mapping_buffer.values[starts[r]:starts[r+1]],expected_context)
        self.assertEqual(self.launcher.calls[0]['grid'],(8,1))

    def test_backend_lifecycle_pending_then_initialized_at_first_inputs(self):
        rows=geometry();draft=rows[0]
        draft.block_size=-1;draft.kv_cache_gid=-1;draft.draft_attn_groups=[]
        self.assertFalse(self.module._validate_backend(draft,allow_pending=True))
        with self.assertRaisesRegex(RuntimeError,'initialized'):self.invoke(rows)
        self.assertEqual(self.launcher.calls,[])
        draft.block_size=560;draft.kv_cache_gid=2;draft.draft_attn_groups=[NS(kv_cache_group_id=2)]
        self.assertTrue(self.module._validate_backend(draft))
        self.invoke(rows)
        self.assertEqual(len(self.launcher.calls),1)

    def test_partial_or_mismatched_backend_initialization_fails_closed(self):
        for page,gid,groups in [(560,-1,[]),(-1,2,[]),(560,2,[]),(560,2,[NS(kv_cache_group_id=3)])]:
            draft=geometry()[0];draft.block_size=page;draft.kv_cache_gid=gid;draft.draft_attn_groups=groups
            with self.assertRaisesRegex(RuntimeError,'initialized'):
                self.module._validate_backend(draft,allow_pending=True)

    def source_modules(self):
        def native():pass
        native.__module__='vllm.v1.spec_decode.dflash'
        native.__qualname__='DFlashProposer.set_inputs_first_pass'
        def original_inputs():pass
        def current():return original_inputs
        class Proxy:
            def __init__(self,original):self.original=original
        kernel=object()
        dflash=NS(__file__=str(NATIVE/'sozo-installed-dflash-as-read.py'),
                  DFlashProposer=NS(set_inputs_first_pass=current),
                  copy_and_expand_dflash_inputs_kernel=Proxy(kernel))
        utils=NS(__file__=str(NATIVE/'sozo-installed-spec-utils-as-read.py'),
                 copy_and_expand_dflash_inputs_kernel=kernel)
        slot=NS(__file__=str(PLUGIN/'dflash_spec.py'),_upstream_set_inputs_first_pass=native,
                set_inputs_first_pass=original_inputs)
        boundary=NS(__file__=str(PLUGIN/'dflash_conv_boundary.py'),_INSTALLED=True,_InputKernelProxy=Proxy)
        paths=lambda fn:dflash.__file__ if fn is native else boundary.__file__
        return dflash,utils,slot,boundary,paths

    def test_source_contract_accepts_exact_native_and_hook_chain(self):
        dflash,utils,slot,boundary,paths=self.source_modules()
        with patch.object(self.module.inspect,'getsourcefile',side_effect=paths):
            records=self.module._source_contract(dflash,utils,slot,boundary)
        self.assertEqual(len(records),4)
        self.assertEqual(records['dflash']['sha256'],self.module.NATIVE_SOURCE_SHA256['dflash'])

    def test_source_contract_rejects_unbound_source_and_unknown_proxy(self):
        dflash,utils,slot,boundary,paths=self.source_modules()
        with patch.object(self.module,'_sha',return_value='0'*64):
            with self.assertRaisesRegex(RuntimeError,'source differs'):
                self.module._source_contract(dflash,utils,slot,boundary)
        dflash.copy_and_expand_dflash_inputs_kernel.original=object()
        with patch.object(self.module.inspect,'getsourcefile',side_effect=paths):
            with self.assertRaisesRegex(RuntimeError,'boundary and per-group'):
                self.module._source_contract(dflash,utils,slot,boundary)

    def test_instance_install_before_backend_init_is_idempotent_and_bound(self):
        rows=geometry();draft=rows[0]
        class Proposer:
            def set_inputs_first_pass(self,*args):return rows[5]
        value=Proposer();value.__dict__.update(draft.__dict__)
        value.block_size=-1;value.kv_cache_gid=-1;value.draft_attn_groups=[]
        value.is_dflash2=True;value.uses_mrope=False
        dflash=NS(DFlashProposer=Proposer)
        spec_module=ModuleType('vllm.v1.spec_decode');spec_module.dflash=dflash;spec_module.utils=NS()
        ornith=ModuleType('ornith_g256');ornith.dflash_spec=NS();ornith.dflash_conv_boundary=NS()
        constants=ModuleType('vllm.v1.attention.backends.utils');constants.NULL_BLOCK_ID=0;constants.PAD_SLOT_ID=-1
        config=NS(speculative_config=NS(method='dflash',num_speculative_tokens=15),use_v2_model_runner=False,
                  scheduler_config=NS(async_scheduling=False,max_num_seqs=8))
        worker=NS(model_runner=NS(drafter=value),vllm_config=config)
        with patch.dict(sys.modules,{'vllm.v1.spec_decode':spec_module,'ornith_g256':ornith,
                        'vllm.v1.attention.backends.utils':constants}), \
                patch.object(self.module,'_source_contract',return_value={'source':'exact'}):
            receipt=self.module.install(worker)
            self.assertFalse(receipt['backend_initialized_at_install'])
            self.assertIs(self.module.install(worker),receipt)
            value.block_size=560;value.kv_cache_gid=2;value.draft_attn_groups=[NS(kv_cache_group_id=2)]
            value.set_inputs_first_pass(rows[2],None,rows[3],None,None,rows[1],None)
            self.assertEqual(len(self.launcher.calls),1)
            value.set_inputs_first_pass=MethodType(Proposer.set_inputs_first_pass,value)
            with self.assertRaisesRegex(RuntimeError,'binding changed'):self.module.install(worker)

    def test_request_shape_or_per_group_alias_drift_fails_before_launch(self):
        rows=geometry()
        rows[5][2].slot_mapping.pointer=-1
        with self.assertRaisesRegex(RuntimeError,'does not alias'): self.invoke(rows)
        self.assertEqual(self.launcher.calls,[])

    def test_bad_rejected_metadata_is_rejected_without_readback(self):
        rows=geometry(rejected_count=1)
        rows[4].dtype='i64'
        with self.assertRaisesRegex(RuntimeError,'rejected-tail'): self.invoke(rows)
        self.assertEqual(self.launcher.calls,[])

    def test_unsupported_width_and_layout_fail_closed(self):
        for change in ('width','stride','device'):
            rows=geometry()
            if change=='width': rows[0].num_speculative_tokens=2
            if change=='stride': rows[1].query_start_loc.strides=(2,)
            if change=='device': rows[0]._context_slot_mapping_buffer.device=NS(type='cpu')
            with self.assertRaises(RuntimeError): self.invoke(rows)
        self.assertEqual(self.launcher.calls,[])

    def test_runtime_context_count_is_not_an_exact_shape_constexpr(self):
        fn=next(n for n in ast.parse(ADAPTER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_physical_slots')
        arg=next(a for a in fn.args.args if a.arg=='context_size')
        self.assertIsNone(arg.annotation)

    def test_k0_context_maintenance_then_k3_reentry_uses_same_hook(self):
        draft,cad,tokens,positions,_,result=geometry(width=8)
        stored=[];native=[]
        class Proposer:
            def set_inputs_first_pass(self,*args):
                width=1+self.num_speculative_tokens
                view=Tensor((width,),pointer=self._slot_mapping_buffer.pointer)
                self._per_group_slot_mappings[self.kv_cache_gid]=view
                self._dflash_num_context=tokens.shape[0]
                self._dflash_hidden_states=args[3]
                self._context_positions_buffer=positions
                return width,object(),NS(slot_mapping=view)
            def propose(self,width,*args,**kwargs):
                self.num_speculative_tokens=width;native.append(kwargs)
                self.set_inputs_first_pass(kwargs['target_token_ids'],kwargs['next_token_ids'],kwargs['target_positions'],
                    kwargs['target_hidden_states'],kwargs['token_indices_to_sample'],kwargs['common_attn_metadata'],
                    kwargs.get('num_rejected_tokens_gpu'))
                return 'native-proposals'
        value=Proposer();value.__dict__.update(draft.__dict__)
        value.model=NS(combine_hidden_states=lambda hidden:hidden,
                       precompute_and_store_context_kv=lambda *args:stored.append(args))
        previous=value.set_inputs_first_pass
        value.set_inputs_first_pass=MethodType(self.module._wrap_inputs(value,previous),value)
        native_module=ModuleType('vllm.v1.spec_decode.dflash');native_module.DFlashProposer=Proposer
        with patch.dict(sys.modules,{'vllm.v1.spec_decode.dflash':native_module,'torch':TORCH}):
            retained=load(PLUGIN/'no_spec.py','cpu_no_spec');retained.install()
        kwargs=dict(target_token_ids=tokens,next_token_ids=Tensor((1,),[5]),target_positions=positions,
                    target_hidden_states=object(),token_indices_to_sample=object(),common_attn_metadata=cad)
        result=value.propose(0,**kwargs)
        self.assertEqual(result.shape,(1,0));self.assertEqual(value.num_speculative_tokens,0)
        self.assertEqual(len(stored),1);self.assertEqual(len(native),0)
        self.assertEqual(stored[0][2].values[:4],value._context_slot_mapping_buffer.values[:4])
        self.assertEqual(value.propose(3,**kwargs),'native-proposals')
        self.assertEqual(len(native),1);self.assertIs(native[0]['target_positions'],positions)
        self.assertEqual([r['width'] for r in self.launcher.calls],[8,4])

    def test_worker_delta_is_only_registration_before_compile(self):
        old=ast.parse((PARENT/'iu4_correctness/worker.py').read_text())
        new=ast.parse((PLUGIN/'iu4_correctness/worker.py').read_text())
        fn=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name=='load_model')
        indexes=[i for i,n in enumerate(fn.body) if isinstance(n,ast.ImportFrom) and n.module=='dflash_physical_slots']
        self.assertEqual(len(indexes),1);i=indexes[0]
        self.assertEqual(ast.unparse(fn.body[i+1]),'self._iu4_physical_slots = install_physical_slots(self, self._iu4_correctness_audit)')
        self.assertIn('mkdir',ast.unparse(fn.body[i-1]))
        self.assertIsInstance(fn.body[i+2],ast.ImportFrom)
        self.assertEqual(fn.body[i+2].module,'final_router_ablation')
        del fn.body[i:i+2]
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_allocator_sampling_k0_and_prefix_guards_are_byte_identical(self):
        for name in ['no_spec.py','prefix_cache.py','cache_full1120.py','dflash_spec.py','dflash_conv_boundary.py',
                     'optimized/greedy.py','iu4_correctness/adaptive_concurrent_k0.py',
                     'iu4_correctness/adaptive_k3_geometry.py','iu4_correctness/adaptive_fallback_policy.py',
                     'iu4_correctness/final_sampler.py','iu4_correctness/final_native_sampling.py',
                     'iu4_correctness/prefill_service.py']:
            old = json.loads((PARENT/'iu4_correctness/source-pins.json').read_text())['files']
            self.assertEqual(old[name],hashlib.sha256((PLUGIN/name).read_bytes()).hexdigest(),name)


if __name__=='__main__': unittest.main()
