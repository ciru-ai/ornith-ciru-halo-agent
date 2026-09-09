"""Parent32tile semantics with four output-column workgroups per KV head."""
# Copyright2026 Ciru.
from collections import Counter
import torch
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend,RocmAttentionImpl
KERNEL_SHA256='27fb053d970184732a46212026afd3152ec07a0dfc3dc4c5cc5cc0d539f17c81'
from .column_kernel import ornith_column_paged_attention


class OrnithColumnAttentionBackend(RocmAttentionBackend):
    @staticmethod
    def get_name(): return 'CUSTOM'
    @staticmethod
    def get_impl_cls(): return OrnithColumnAttentionImpl


class OrnithColumnAttentionImpl(RocmAttentionImpl):
    implementation='ciru.ornith.column.backend.v1'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dispatch_counts=Counter()
        self.capture_by_C=Counter()
        self.eager_by_C=Counter()
        self.context_bounds=[None,None]

    def static_native_support(self):
        return (self.attn_type==AttentionType.DECODER
            and (self.num_heads,self.num_kv_heads,self.head_size)==(16,2,256)
            and self.scale==.0625 and self.kv_cache_dtype in ('auto','bfloat16')
            and self.alibi_slopes is None and self.sliding_window==(-1,-1)
            and self.logits_soft_cap==0 and self.sinks is None
            and self.kv_sharing_target_layer_name is None)

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=None,output_block_scale=None):
        m=attn_metadata;reason=None
        if m is None: reason='profile'
        elif not self.static_native_support(): reason='static_feature'
        elif (m.use_cascade or m.causal is not True or output_scale is not None
              or output_block_scale is not None): reason='metadata_feature'
        elif m.max_query_len!=1: reason='prefill_or_mixed'
        else:
            C=m.seq_lens.shape[0]
            if (C<1 or query.shape!=(C,16,256) or output.shape!=query.shape
                    or not 0<m.num_actual_tokens<=C or m.block_table.shape[0]!=C
                    or m.query_start_loc.shape!=(C+1,)):
                reason='query_mapping'
            elif (query.dtype!=torch.bfloat16 or output.dtype!=torch.bfloat16
                    or kv_cache.dtype!=torch.bfloat16 or m.block_table.dtype!=torch.int32
                    or m.seq_lens.dtype!=torch.int32 or m.query_start_loc.dtype!=torch.int32):
                reason='dtype'
            elif (query.stride(2)!=1 or output.stride(2)!=1 or m.block_table.stride(1)!=1
                    or not m.seq_lens.is_contiguous() or not m.query_start_loc.is_contiguous()):
                reason='stride'
            else:
                kc,vc=PagedAttention.split_kv_cache(kv_cache.transpose(0,1),2,256)
                if vc.shape[3]!=1056: reason='cache_page_size'
                else:
                    # P4's ordinary opaque Attention op owns current-stream
                    # execution and its preceding inherited cache update.
                    ornith_column_paged_attention[(C,2,4)](output,query,kc,vc,None,m.block_table,
                        m.seq_lens,None,self.scale,layer._k_scale,layer._v_scale,1.,
                        num_query_heads=16,num_queries_per_kv=8,num_queries_per_kv_padded=16,
                        block_table_stride=m.block_table.stride(0),query_stride_0=query.stride(0),
                        query_stride_1=query.stride(1),output_stride_0=output.stride(0),
                        output_stride_1=output.stride(1),BLOCK_SIZE=32,PHYSICAL_BLOCK_SIZE=1056,
                        HEAD_SIZE=256,HEAD_SIZE_PADDED=256,USE_ALIBI_SLOPES=False,SLIDING_WINDOW=0,x=8,
                        stride_k_cache_0=kc.stride(0),stride_k_cache_1=kc.stride(1),
                        stride_k_cache_2=kc.stride(2),stride_k_cache_3=kc.stride(3),stride_k_cache_4=kc.stride(4),
                        stride_v_cache_0=vc.stride(0),stride_v_cache_1=vc.stride(1),
                        stride_v_cache_2=vc.stride(2),stride_v_cache_3=vc.stride(3),
                        filter_by_query_len=True,query_start_len_ptr=m.query_start_loc,USE_SINKS=False,USE_FP8=False)
                    capturing=torch.cuda.is_current_stream_capturing()
                    self.dispatch_counts['capture_native_calls' if capturing else 'eager_native_calls']+=1
                    (self.capture_by_C if capturing else self.eager_by_C)[str(C)]+=1
                    self.dispatch_counts['native_calls']+=1
                    if not capturing and m.max_seq_len>1:self.dispatch_counts['eager_cached_native_calls']+=1
                    lo,hi=self.context_bounds
                    self.context_bounds=[m.max_seq_len if lo is None else min(lo,m.max_seq_len),
                                         m.max_seq_len if hi is None else max(hi,m.max_seq_len)]
                    return output
        self.dispatch_counts['fallback_'+reason]+=1
        return super().forward(layer,query,key,value,kv_cache,m,output,output_scale,output_block_scale)

    def inspect_dispatch(self,prefix=None):
        return {'implementation':self.implementation,'kernel_sha256':KERNEL_SHA256,'prefix':prefix,
                'counts':dict(self.dispatch_counts),'capture_by_C':dict(self.capture_by_C),
                'eager_by_C':dict(self.eager_by_C),'native_max_seq_len_host_bounds':list(self.context_bounds),
                'counter_semantics':'Host dispatch counts; capture keyed by actual C/prefix, graph replay counted separately',
                'capacity_policy':'inherits parent context/batch capacity; tested coverage recorded externally'}
