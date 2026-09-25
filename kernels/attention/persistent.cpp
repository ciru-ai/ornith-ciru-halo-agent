// Copyright 2026 Ciru. Fixed-shape cached signed IU4 attention component.
// Optional padded page1120 persistent IU4; GPU metadata/long-only dispatch. QK D256 once, fused two PV D128 halves; packed P/V nonDC scale and FP16 DC.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>
using i32x2=int32_t __attribute__((ext_vector_type(2)));
using i32x8=int32_t __attribute__((ext_vector_type(8)));
using bf16=hip_bfloat16;
constexpr int Q=64,D=256,PAGE=1120,GROUPS=PAGE/32,SPLITS=32,PARTS=33;
// Physical BF16 page stride includes 98,304 bytes of full1120 padding.
constexpr size_t PAGE_STRIDE=1196032;
__device__ __forceinline__ bf16 cache_get(const bf16* cache,const int* table,int t,int kv,int h,int d){
    return cache[size_t(table[t/PAGE])*PAGE_STRIDE+(kv*PAGE+t%PAGE)*512+h*256+d];
}
__device__ __forceinline__ float scale_for(float a){
    return a==0 ? 1.f : __half2float(__float2half_rn(fmaxf(a/7.f,0x1p-14f)));
}
__device__ __forceinline__ uint32_t code(float x,float s){
    int v=int(rintf(x/s)); v=v< -7? -7:(v>7?7:v);return uint32_t(v)&15;
}
__device__ __forceinline__ uint32_t pack_vmeta(float scale,float dc){
    return uint32_t(__half_as_ushort(__float2half_rn(scale))) |
           (uint32_t(__half_as_ushort(__float2half_rn(dc)))<<16);
}
__device__ __forceinline__ float meta_scale(uint32_t meta){
    return __half2float(__ushort_as_half(uint16_t(meta)));
}
__device__ __forceinline__ float meta_dc(uint32_t meta){
    return __half2float(__ushort_as_half(uint16_t(meta>>16)));
}
template<bool IsQ> __global__ __launch_bounds__(256) void prep_qk(
        const bf16* src,uint32_t* packed,__half* scales,int tokens,int stride0,int stride1,const int* dirty,int count,const int* owners=nullptr,const int* seq=nullptr,int threshold=0){
    constexpr int Heads=IsQ?16:2;
    int lane=threadIdx.x&31,row=blockIdx.x*8+(threadIdx.x>>5);
    int group=IsQ?0:dirty[blockIdx.x/8];
    if constexpr(IsQ){if(row>=Heads*tokens)return;int owner=owners[row/16];if(owner<0 || seq[owner]<threshold)return;}
    else {if(group<0)return; row=(blockIdx.x%8)*8+(threadIdx.x>>5);}
    int t=row/Heads,h=row%Heads;float x[8];
    size_t slot=IsQ?0:size_t(group/GROUPS)*PAGE+(group%GROUPS)*32+t;
#pragma unroll
    for(int j=0;j<8;++j)x[j]=float(IsQ?src[size_t(t)*stride0+h*stride1+lane*8+j]:src[(slot/PAGE)*PAGE_STRIDE+(slot%PAGE)*512+h*256+lane*8+j]);
    // Normalized H256: three register and five wave butterflies.
#pragma unroll
    for(int stride=1;stride<8;stride*=2){
#pragma unroll
        for(int j=0;j<8;++j)if((j&stride)==0){float a=x[j],b=x[j+stride];x[j]=a+b;x[j+stride]=a-b;}
    }
#pragma unroll
    for(int stride=1;stride<32;stride*=2){
#pragma unroll
        for(int j=0;j<8;++j){float peer=__shfl_xor(x[j],stride,32);x[j]=(lane&stride)?peer-x[j]:x[j]+peer;}
    }
    float a=0;
#pragma unroll
    for(int j=0;j<8;++j){x[j]*=.0625f;a=fmaxf(a,fabsf(x[j]));}
#pragma unroll
    for(int stride=16;stride;stride/=2)a=fmaxf(a,__shfl_xor(a,stride,32));
    float s=scale_for(a);uint32_t word=0;
#pragma unroll
    for(int j=0;j<8;++j)word|=code(x[j],s)<<(4*j);
    if constexpr(IsQ){packed[(size_t(h)*Q+t)*32+lane]=word;if(lane==0)scales[h*Q+t]=__float2half_rn(s);}
    else {packed[((size_t(group)*2+h)*32+lane)*32+t]=word;if(lane==0)scales[(size_t(group)*2+h)*32+t]=__float2half_rn(s);}
}
__global__ __launch_bounds__(256) void prep_v(const bf16* src,uint32_t* packed,uint32_t* scales,const int* dirty,int count,int* busy=nullptr,int* valid=nullptr){
    int block=dirty[blockIdx.x],h=blockIdx.y,col=threadIdx.x;
    if(block<0)return;
    size_t slot=size_t(block/GROUPS)*PAGE+(block%GROUPS)*32;
    float x[32];
#pragma unroll
    for(int j=0;j<32;++j)x[j]=float(src[(slot/PAGE)*PAGE_STRIDE+(PAGE+slot%PAGE+j)*512+h*256+col]);
#pragma unroll
    for(int stride=1;stride<32;stride*=2){
#pragma unroll
        for(int j=0;j<32;++j)if((j&stride)==0){float a=x[j],b=x[j+stride];x[j]=a+b;x[j+stride]=a-b;}
    }
    float a=0;
#pragma unroll
    for(int j=0;j<32;++j){x[j]*=0.1767766952966369f;if(j)a=fmaxf(a,fabsf(x[j]));}
    float dc=x[0];x[0]=0;
    float s=scale_for(a);scales[(size_t(block)*2+h)*256+col]=pack_vmeta(s,dc);
#pragma unroll
    for(int w=0;w<4;++w){uint32_t word=0;
#pragma unroll
        for(int j=0;j<8;++j)word|=code(x[w*8+j],s)<<(4*j);
        packed[((size_t(block)*2+h)*4+w)*256+col]=word;
    }
    if(h==0 && col==0){if(busy)busy[block]=0;if(valid)valid[block]=1;}
}
// Mark each physical group once using a reusable GPU flag, then prep_v clears it.
__global__ void dirty_groups(const int64_t* slots,int* groups,int* busy,int rows){
    int row=blockIdx.x*256+threadIdx.x;
    if(row>=rows)return;
    int64_t slot=slots[row];int group=slot<0?-1:int(slot/32);
    if(group>=0 && atomicCAS(busy+group,0,1)!=0)group=-1;
    groups[row]=group;
}
__global__ void metadata(const int* starts,const int* seq,int* contexts,int* counts,
        int* owners,int requests,int rows,int threshold,unsigned long long* stats){
    int i=threadIdx.x;
    if(i<requests){
        int count=starts[i+1]-starts[i];contexts[i]=seq[i]-count;counts[i]=count;
        if(count>0){atomicAdd(stats+(seq[i]>=threshold?0:1),1ull);
            if(seq[i]>=threshold)atomicAdd(stats+2,(unsigned long long)count);}
    }
    if(i<rows){int owner=-1;
        for(int r=0;r<requests;++r)if(i>=starts[r] && i<starts[r+1])owner=r;
        owners[i]=owner;
    }
}
__global__ __launch_bounds__(256) void attention(const uint32_t* qp,const __half* qs,
        const uint32_t* kp,const __half* ks,const uint32_t* vp,const uint32_t* vs,float* out,
        const int* table,const int* contexts,const int* counts,const int* starts,
        const int* seq,int cols,int query_tiles,int threshold){
    __shared__ uint32_t sk[32*32],sv[4*256];
    __shared__ float sks[32];
    __shared__ uint32_t svmeta[256],spp[64*4];
    __shared__ float salpha[64],spscale[64],spdc[64],snorm[64];
    int tid=threadIdx.x,lane=tid&31,row16=lane&15,piece=lane>>4,wave=(tid>>5)&3;
    int req=blockIdx.x/query_tiles,qtile=blockIdx.x%query_tiles;
    int query_count=counts[req];
    if(seq[req]<threshold || query_count<=qtile*8)return;
    int kh=blockIdx.y,dhalf=tid/128,split=blockIdx.z,folded=wave*16+row16;
    int row=starts[req]+qtile*8+folded/8,h=kh*8+folded%8;
    bool live=qtile*8+folded/8<query_count;float sq=live && dhalf==0?__half2float(qs[h*Q+row]):1.f;
    // Each query chooses its own precision boundary; keep at least32 recent
    // tokens in BF16 regardless of speculative batch start or accepted depth.
    int query_pos=contexts[req]+qtile*8+folded/8;
    int row_hist=max(0,(query_pos-32)/32);
    int hist=max(0,(contexts[req]+query_count-1-32)/32);
    // Fixed blocks of64 groups preserve the public partition order through64K.
    // Further blocks use a fixed2048-group stride, independent of batch maximum.
    int begin=split*64,end=hist;
    if(begin>=end){
        if(live){
#pragma unroll
            for(int j=0;j<64;++j)out[((size_t(row)*16+h)*PARTS+split)*257+dhalf*128+2*j+piece]=0.f;
            if(piece==0 && dhalf==0)out[((size_t(row)*16+h)*PARTS+split)*257+256]=-INFINITY;
        }
        return;
    }
    uint32_t q[32];
    if(dhalf==0){
#pragma unroll
        for(int j=0;j<32;++j)q[j]=live?qp[(size_t(h)*Q+row)*32+j]:0;
    }
    float accum[64]={},m=-INFINITY,l=0;
    // All lanes participate in barriers, including the final CTA query tail.
    for(int block=begin;block<end;block+=((block&63)==63 ? (SPLITS-1)*64+1 : 1)){
        size_t pg=size_t(table[req*cols+block/GROUPS])*GROUPS+block%GROUPS;
        size_t bank=pg*2+kh;
#pragma unroll
        for(int j=0;j<4;++j){int i=tid+256*j;sk[i]=kp[bank*1024+i];}
#pragma unroll
        for(int j=0;j<4;++j){int i=tid+256*j;sv[i]=vp[bank*1024+i];}
        if(tid<32)sks[tid]=__half2float(ks[bank*32+tid]);
        svmeta[tid]=vs[bank*256+tid];
        __syncthreads();
        uint32_t pp[4]={};float alpha,sp,pdc;
        if(dhalf==0){
        float p[16];float old_m=m,old_l=l;
#pragma unroll
        for(int tile=0;tile<2;++tile){
            i32x8 dots={};
#pragma unroll
            for(int step=0;step<16;++step){
                i32x2 a={int32_t(sk[(2*step)*32+tile*16+row16]),int32_t(sk[(2*step+1)*32+tile*16+row16])};
                i32x2 b={int32_t(q[2*step]),int32_t(q[2*step+1])};
                dots=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true,a,true,b,dots,false);
            }
#pragma unroll
            for(int j=0;j<8;++j){int key=tile*16+2*j+piece;
                float score=float(dots[j])*sq*sks[key]*.0625f;
                p[tile*8+j]=score;
            }
        }
        float mx=-INFINITY;
#pragma unroll
        for(int j=0;j<16;++j)mx=fmaxf(mx,p[j]);
        mx=fmaxf(mx,__shfl_xor(mx,16,32));
        float next=fmaxf(m,mx),sum=0;alpha=exp2f((m-next)*1.4426950408889634f);
#pragma unroll
        for(int j=0;j<16;++j){p[j]=exp2f((p[j]-next)*1.4426950408889634f);sum+=p[j];}
        sum+=__shfl_xor(sum,16,32);l=l*alpha+sum;m=next;
        // P H32: dimension bit0 is split across wave halves.
#pragma unroll
        for(int j=0;j<16;++j){float peer=__shfl_xor(p[j],16,32);p[j]=piece?peer-p[j]:p[j]+peer;}
#pragma unroll
        for(int stride=1;stride<16;stride*=2){
#pragma unroll
            for(int j=0;j<16;++j)if((j&stride)==0){float a=p[j],b=p[j+stride];p[j]=a+b;p[j+stride]=a-b;}
        }
        float amax=0;
#pragma unroll
        for(int j=0;j<16;++j)p[j]*=0.1767766952966369f;
        pdc=__shfl(p[0],row16,32);if(piece==0)p[0]=0;
#pragma unroll
        for(int j=0;j<16;++j)amax=fmaxf(amax,fabsf(p[j]));
        amax=fmaxf(amax,__shfl_xor(amax,16,32));sp=scale_for(amax);

#pragma unroll
        for(int w=0;w<4;++w){
#pragma unroll
            for(int j=0;j<4;++j){uint32_t own=code(p[w*4+j],sp),peer=__shfl_xor(own,16,32);
                pp[w]|=(piece?peer:own)<<(8*j);pp[w]|=(piece?own:peer)<<(8*j+4);
            }
        }
        if(block>=row_hist){
            alpha=1.f;sp=0.f;pdc=0.f;l=old_l;m=old_m;
#pragma unroll
            for(int w=0;w<4;++w)pp[w]=0;
        }
        if(piece==0){
#pragma unroll
            for(int w=0;w<4;++w)spp[folded*4+w]=pp[w];
            salpha[folded]=alpha;spscale[folded]=sp;spdc[folded]=pdc;snorm[folded]=l;
        }
        }
        __syncthreads();
        if(dhalf!=0){
#pragma unroll
            for(int w=0;w<4;++w)pp[w]=spp[folded*4+w];
            alpha=salpha[folded];sp=spscale[folded];pdc=spdc[folded];
        }
#pragma unroll
        for(int j=0;j<64;++j)accum[j]*=alpha;
#pragma unroll
        for(int tile=0;tile<8;++tile){i32x8 dots={};
#pragma unroll
            for(int step=0;step<2;++step){
                i32x2 a={int32_t(sv[(2*step)*256+dhalf*128+tile*16+row16]),int32_t(sv[(2*step+1)*256+dhalf*128+tile*16+row16])};
                i32x2 b={int32_t(pp[2*step]),int32_t(pp[2*step+1])};
                dots=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true,a,true,b,dots,false);
            }
#pragma unroll
            for(int j=0;j<8;++j){
                uint32_t meta=svmeta[dhalf*128+tile*16+2*j+piece];
                accum[tile*8+j]=fmaf(pdc,meta_dc(meta),
                    accum[tile*8+j]+float(dots[j])*sp*meta_scale(meta));
            }
        }
        __syncthreads();
    }
    if(dhalf!=0)l=snorm[folded];
    if(live){
#pragma unroll
        for(int j=0;j<64;++j)out[((size_t(row)*16+h)*PARTS+split)*257+dhalf*128+2*j+piece]=l>0.f?accum[j]/l:0.f;
        if(piece==0 && dhalf==0)out[((size_t(row)*16+h)*PARTS+split)*257+256]=m+logf(l);
    }
}

// Exact BF16 operands and FP32 reduction over only the causal boundary tail.
__global__ __launch_bounds__(128) void tail(const bf16* q,const bf16* cache,const int* table,const int* contexts,const int* starts,const int* owners,
        const int* seq,float* parts,int rows,int cols,int q_stride0,int q_stride1,int threshold){
    int lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows*16)return;
    int qr=row/16,h=row%16,req=owners[qr];
    if(req<0 || seq[req]<threshold)return;
    int context=contexts[req];table+=req*cols;
    int last=context+qr-starts[req],first=max(0,(last-32)/32)*32;
    float query[8],acc[8]={},m=-INFINITY,l=0;
#pragma unroll
    for(int j=0;j<8;j++)query[j]=float(q[size_t(qr)*q_stride0+h*q_stride1+lane*8+j]);
    for(int t=first;t<=last;t++){
        float dot=0;
#pragma unroll
        for(int j=0;j<8;j++)dot+=query[j]*float(cache_get(cache,table,t,0,h/8,lane*8+j));
#pragma unroll
        for(int k=16;k;k/=2)dot+=__shfl_xor(dot,k,32);
        dot*=.0625f;float next=fmaxf(m,dot),alpha=expf(m-next),p=expf(dot-next);l=l*alpha+p;m=next;
#pragma unroll
        for(int j=0;j<8;j++)acc[j]=acc[j]*alpha+p*float(cache_get(cache,table,t,1,h/8,lane*8+j));
    }
#pragma unroll
    for(int j=0;j<8;j++)parts[(size_t(row)*PARTS+32)*257+lane*8+j]=acc[j]/l;
    if(lane==0)parts[(size_t(row)*PARTS+32)*257+256]=m+logf(l);
}
__global__ __launch_bounds__(128) void reduce(const float* parts,bf16* out,const int* owners,const int* seq,int rows,int stride0,int stride1,int threshold){
    int lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows*16)return;
    int qr=row/16,h=row%16,req=owners[qr];
    if(req<0){
#pragma unroll
        for(int j=0;j<8;j++)out[size_t(qr)*stride0+h*stride1+lane*8+j]=bf16(0.f);
        return;
    }
    if(seq[req]<threshold)return;
    float m=-INFINITY,l=0,acc[8]={};
    for(int p=0;p<PARTS;p++)m=fmaxf(m,parts[(size_t(row)*PARTS+p)*257+256]);
    for(int p=0;p<PARTS;p++){
        float w=expf(parts[(size_t(row)*PARTS+p)*257+256]-m);l+=w;
#pragma unroll
        for(int j=0;j<8;j++)acc[j]+=w*parts[(size_t(row)*PARTS+p)*257+lane*8+j];
    }
#pragma unroll
    for(int j=0;j<8;j++)out[size_t(qr)*stride0+h*stride1+lane*8+j]=bf16(acc[j]/l);
}
// Lifecycle payload copies are restricted to groups populated by target writers.
__global__ void reset_pages(int* valid,int* busy,const int64_t* pages,int count){
    int index=blockIdx.x*256+threadIdx.x;
    if(index<count*GROUPS){int group=int(pages[index/GROUPS])*GROUPS+index%GROUPS;valid[group]=0;busy[group]=0;}
}
__global__ void copy_pages(uint32_t* kp,__half* ks,uint32_t* vp,uint32_t* vs,
        int* valid,int* busy,const int64_t* pairs,int count){
    int pair=blockIdx.x/GROUPS,g=blockIdx.x%GROUPS,tid=threadIdx.x;
    if(pair>=count)return;
    size_t src=size_t(pairs[pair*2])*GROUPS+g,dst=size_t(pairs[pair*2+1])*GROUPS+g;
    int live=valid[src];
    if(live){
        for(int i=tid;i<2048;i+=256){kp[dst*2048+i]=kp[src*2048+i];vp[dst*2048+i]=vp[src*2048+i];}
        for(int i=tid;i<512;i+=256)vs[dst*512+i]=vs[src*512+i];
        if(tid<64)ks[dst*64+tid]=ks[src*64+tid];
    }
    if(tid==0){valid[dst]=live;busy[dst]=0;}
}
extern "C" __attribute__((visibility("default"))) int update_cache(const void* cache,const void* slots,void* groups,void* busy,void* valid,void* kp,void* ks,void* vp,void* vs,int rows,void* stream){
    if(rows==0)return 0;auto s=reinterpret_cast<hipStream_t>(stream);
    dirty_groups<<<(rows+255)/256,256,0,s>>>((const int64_t*)slots,(int*)groups,(int*)busy,rows);
    prep_qk<false><<<rows*8,256,0,s>>>((const bf16*)cache,(uint32_t*)kp,(__half*)ks,0,512,256,(const int*)groups,rows);
    prep_v<<<dim3(rows,2),256,0,s>>>((const bf16*)cache,(uint32_t*)vp,(uint32_t*)vs,(const int*)groups,rows,(int*)busy,(int*)valid);
    return int(hipGetLastError());
}
extern "C" __attribute__((visibility("default"))) int prepare_decode(const void* q,const void* starts,const void* seq,void* contexts,void* counts,void* owners,void* stats,void* qp,void* qs,int requests,int rows,int stride0,int stride1,int threshold,void* stream){
    auto s=reinterpret_cast<hipStream_t>(stream);
    metadata<<<1,128,0,s>>>((const int*)starts,(const int*)seq,(int*)contexts,(int*)counts,(int*)owners,requests,rows,threshold,(unsigned long long*)stats);
    prep_qk<true><<<(rows*16+7)/8,256,0,s>>>((const bf16*)q,(uint32_t*)qp,(__half*)qs,rows,stride0,stride1,nullptr,0,(const int*)owners,(const int*)seq,threshold);
    return int(hipGetLastError());
}
extern "C" __attribute__((visibility("default"))) int cached_attention(const void* qp,const void* qs,const void* kp,const void* ks,const void* vp,const void* vs,void* parts,const void* table,const void* contexts,const void* counts,const void* starts,const void* seq,int requests,int cols,int query_tiles,int threshold,void* stream){
    attention<<<dim3(requests*query_tiles,2,32),256,0,reinterpret_cast<hipStream_t>(stream)>>>((const uint32_t*)qp,(const __half*)qs,(const uint32_t*)kp,(__half*)ks,(const uint32_t*)vp,(const uint32_t*)vs,(float*)parts,(const int*)table,(const int*)contexts,(const int*)counts,(const int*)starts,(const int*)seq,cols,query_tiles,threshold);
    return int(hipGetLastError());
}
extern "C" __attribute__((visibility("default"))) int tail_reduce(const void* q,const void* cache,const void* table,const void* contexts,const void* starts,const void* owners,const void* seq,void* parts,void* out,int rows,int cols,int qs0,int qs1,int os0,int os1,int threshold,void* stream){
    auto s=reinterpret_cast<hipStream_t>(stream);
    tail<<<(rows*16+3)/4,128,0,s>>>((const bf16*)q,(const bf16*)cache,(const int*)table,(const int*)contexts,(const int*)starts,(const int*)owners,(const int*)seq,(float*)parts,rows,cols,qs0,qs1,threshold);
    reduce<<<(rows*16+3)/4,128,0,s>>>((const float*)parts,(bf16*)out,(const int*)owners,(const int*)seq,rows,os0,os1,threshold);
    return int(hipGetLastError());
}
extern "C" __attribute__((visibility("default"))) int cache_reset(void* valid,void* busy,const void* pages,int count,void* stream){
    if(count)reset_pages<<<(count*GROUPS+255)/256,256,0,reinterpret_cast<hipStream_t>(stream)>>>((int*)valid,(int*)busy,(const int64_t*)pages,count);
    return int(hipGetLastError());
}
extern "C" __attribute__((visibility("default"))) int cache_copy(void* kp,void* ks,void* vp,void* vs,void* valid,void* busy,const void* pairs,int count,void* stream){
    if(count)copy_pages<<<count*GROUPS,256,0,reinterpret_cast<hipStream_t>(stream)>>>((uint32_t*)kp,(__half*)ks,(uint32_t*)vp,(uint32_t*)vs,(int*)valid,(int*)busy,(const int64_t*)pairs,count);
    return int(hipGetLastError());
}
