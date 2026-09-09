// Full N32 storage: direct T<=16 and grouped/sparse prefill.
// Codes [E,N/32,K/256,32,32], metadata U32[E,N/32,K/256,32].
// Copyright (c) Ciru. Direct decode and grouped prefill routed prototype.
// Arithmetic adapted from qkvz_g128_consumer and grouped_consumer.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "routed_direct_api.h"
#include "portable_exp.h"
namespace routed_direct {
constexpr int E=256,Top=8,H=2048,I=512,G=256,Words=G/8,Parts=G/32;
constexpr float InvSqrt128=0x1.6a09e6p-4f;
struct Meta { __half scale,offset; };
size_t align256(size_t n){return (n+255)&~size_t(255);}
template<class T>T* at(void* p,size_t n){return reinterpret_cast<T*>(static_cast<uint8_t*>(p)+n);}
__device__ __forceinline__ float divide(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ uint16_t bf16_rne(float value){
    unsigned bits=__float_as_uint(value);
    if((bits&0x7fffffffu)>0x7f800000u)return uint16_t((bits>>16)|0x40u);
    return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
template<bool BF16> __global__ __launch_bounds__(128) void transform(const void* input,float* y,uint32_t* sticky,const int32_t* ids){
    size_t base=size_t(blockIdx.x)*128;int lane=threadIdx.x;
    if constexpr(BF16){
        int token=base/H;bool live=false;
#pragma unroll
        for(int slot=0;slot<Top;++slot){int id=ids[token*Top+slot];live|=id>=0&&id<E;}
        if(!live){y[base+lane]=0.f;return;}
    }
    float value;
    if constexpr(BF16)value=__uint_as_float(uint32_t(static_cast<const uint16_t*>(input)[base+lane])<<16);
    else value=static_cast<const float*>(input)[base+lane];
    if(!isfinite(value))atomicOr(sticky,unsigned(1));
    __shared__ float first[128],second[128];first[lane]=value;__syncthreads();
    float* src=first;float* dst=second;
#pragma unroll
    for(int step=1;step<128;step*=2){
        int lo=lane&~step;dst[lane]=(lane&step)?src[lo]-src[lo+step]:src[lo]+src[lo+step];
        __syncthreads();float* swap=src;src=dst;dst=swap;
    }
    float result=src[lane]*InvSqrt128;y[base+lane]=result;
    if(!isfinite(result))atomicOr(sticky,unsigned(1));
}
template<bool A8>__global__ __launch_bounds__(32) void quantize(const float* x,uint32_t* low,uint32_t* high,
        __half* scales,int32_t* sums,uint32_t* sticky){
    int lane=threadIdx.x;size_t group=blockIdx.x;
    float values[Parts],absmax=0.f;
#pragma unroll
    for(int part=0;part<Parts;++part){
        float original=x[group*G+part*32+lane];bool finite=isfinite(original);
        if(!finite)atomicOr(sticky,unsigned(2));
        values[part]=finite?original:0.f;absmax=fmaxf(absmax,fabsf(values[part]));
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)absmax=fmaxf(absmax,__shfl_xor(absmax,delta,32));
    constexpr int qmax=A8?127:7;
    float raw=divide(absmax,float(qmax));bool valid=isfinite(raw)&&raw<=65504.f;
    if(!valid)atomicOr(sticky,unsigned(2));
    __half scale=__float2half_rn(absmax==0.f||!valid?1.f:fmaxf(raw,0x1p-14f));
    int sum=0;
#pragma unroll
    for(int part=0;part<Parts;++part){
        int q=absmax==0.f||!valid?0:__float2int_rn(divide(values[part],__half2float(scale)));
        q=q< -qmax?-qmax:(q>qmax?qmax:q);sum+=q;
        int lo=q&15,hi=(q-lo)/16,first=lane&~7;uint32_t wl=0,wh=0;
#pragma unroll
        for(int d=0;d<8;++d){
            wl|=(unsigned(__shfl(lo,first+d,32))&15u)<<(4*d);
            if constexpr(A8)wh|=(unsigned(__shfl(hi,first+d,32))&15u)<<(4*d);
        }
        if(!(lane&7)){low[group*Words+part*4+lane/8]=wl;if constexpr(A8)high[group*Words+part*4+lane/8]=wh;}
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)sum+=__shfl_xor(sum,delta,32);
    if(lane==0){scales[group]=scale;sums[group]=sum;}
}

template<int N,int K,bool Gate,bool A8> __device__ __forceinline__ void projection_job(
        const int32_t* ids,const uint32_t* codes,const Meta* metadata,
        const uint32_t* low,const uint32_t* high,const __half* scales,const int32_t* sums,
        float* output,uint32_t* sticky,int route,int tile){
    constexpr int Groups=K/G,NTiles=N/16;
    int lane=threadIdx.x,piece=lane>>4,n=tile*16+(lane&15);
    int expert=ids[route];
    if(expert<0 || expert>=E){
        if(expert>=E || expert< -1)atomicOr(sticky,4u);
        if(piece==0)output[size_t(route)*N+n]=0.f;return;
    }
    int row=Gate?route/Top:route;
    size_t abase=size_t(row)*(K/8),wbase=size_t(expert)*N*(K/8),mbase=size_t(expert)*N*Groups;
    float value=0.f;
    for(int group=0;group<Groups;++group){
        int dot=0;
#pragma unroll
        for(int j=0;j<Words/2;++j){
            int local=piece*(Words/2)+j,word=group*Words+local;
            int w=int(codes[wbase+((size_t(n/32)*Groups+group)*Words+local)*32+(n&31)]);
            int lo=int(low[abase+word]);
            if constexpr(A8){
                int hi=int(high[abase+word]);
                int dl=__builtin_amdgcn_sudot8(false,w,false,lo,0,false);
                int dh=__builtin_amdgcn_sudot8(false,w,true,hi,0,false);
                dot+=dl+16*dh;
            }else dot+=__builtin_amdgcn_sudot8(false,w,true,lo,0,false);
        }
        dot=dot+__shfl_xor(dot,16,32);
        if(piece==0){
            float sa=__half2float(scales[size_t(row)*Groups+group]);
            Meta wm=metadata[mbase+(size_t(n/32)*Groups+group)*32+(n&31)];
            float scaled_dot=sa*float(dot),scaled_sum=sa*float(sums[size_t(row)*Groups+group]);
            float product=__half2float(wm.scale)*scaled_dot,correction=__half2float(wm.offset)*scaled_sum;
            value=value+(product+correction);
        }
    }
    if(piece==0){output[size_t(route)*N+n]=value;if(!isfinite(value))atomicOr(sticky,8u);}
}
template<int N,int K,bool Gate,bool A8> __global__ __launch_bounds__(32) void projection(
        const int32_t* ids,const uint32_t* codes,const Meta* metadata,
        const uint32_t* low,const uint32_t* high,const __half* scales,const int32_t* sums,
        float* output,uint32_t* sticky){
    projection_job<N,K,Gate,A8>(ids,codes,metadata,low,high,scales,sums,output,sticky,blockIdx.x/(N/16),blockIdx.x%(N/16));
}
// Experiment-only N32 direct DOT: same integer dot and ascending-group FP32 arithmetic.
template<int N,int K,bool Gate,bool A8> __device__ __forceinline__ void projection_n32_job(
        const int32_t* ids,const uint32_t* codes,const Meta* metadata,
        const uint32_t* low,const uint32_t* high,const __half* scales,const int32_t* sums,
        float* output,uint32_t* sticky,int route,int tile){
    constexpr int Groups=K/G,NTiles=N/32;
    int lane=threadIdx.x,n=tile*32+lane;
    int expert=ids[route];
    if(expert<0 || expert>=E){
        if(expert>=E || expert< -1)atomicOr(sticky,4u);
        output[size_t(route)*N+n]=0.f;return;
    }
    int row=Gate?route/Top:route;
    size_t abase=size_t(row)*(K/8),wbase=size_t(expert)*N*(K/8),mbase=size_t(expert)*N*Groups;
    float value=0.f;
    for(int group=0;group<Groups;++group){
        int dot=0;
#pragma unroll
        for(int j=0;j<Words;++j){
            int local=j,word=group*Words+j;
            int w=int(codes[wbase+((size_t(n/32)*Groups+group)*Words+local)*32+(n&31)]);
            int lo=int(low[abase+word]);
            if constexpr(A8){
                int hi=int(high[abase+word]);
                int dl=__builtin_amdgcn_sudot8(false,w,false,lo,0,false);
                int dh=__builtin_amdgcn_sudot8(false,w,true,hi,0,false);
                dot+=dl+16*dh;
            }else dot+=__builtin_amdgcn_sudot8(false,w,true,lo,0,false);
        }
        {
            float sa=__half2float(scales[size_t(row)*Groups+group]);
            Meta wm=metadata[mbase+(size_t(n/32)*Groups+group)*32+(n&31)];
            float scaled_dot=sa*float(dot),scaled_sum=sa*float(sums[size_t(row)*Groups+group]);
            float product=__half2float(wm.scale)*scaled_dot,correction=__half2float(wm.offset)*scaled_sum;
            value=value+(product+correction);
        }
    }
    {output[size_t(route)*N+n]=value;if(!isfinite(value))atomicOr(sticky,8u);}
}
template<int N,int K,bool Gate,bool A8> __global__ __launch_bounds__(32) void projection_n32(
        const int32_t* ids,const uint32_t* codes,const Meta* metadata,
        const uint32_t* low,const uint32_t* high,const __half* scales,const int32_t* sums,
        float* output,uint32_t* sticky){
    projection_n32_job<N,K,Gate,A8>(ids,codes,metadata,low,high,scales,sums,output,sticky,blockIdx.x/(N/32),blockIdx.x%(N/32));
}
__global__ void middle_kernel(const float* gate,float* middle,uint32_t* sticky,size_t values,const int32_t* ids){
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=values)return;
    size_t route=index/I,column=index%I;
    if(ids[route]<0||ids[route]>=E){middle[index]=0.f;return;}
    float g=gate[route*(2*I)+column],up=gate[route*(2*I)+I+column];
    float denominator=1.f+ornith_portable_exp(-g),silu=divide(g,denominator),result=silu*up;
    middle[index]=result;if(!isfinite(result))atomicOr(sticky,16u);
}
__global__ void reduce_slots(const float* routes,const float* weights,const int32_t* ids,
        float* diagnostic,uint16_t* out,uint32_t* sticky,size_t values){
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=values)return;
    size_t token=index/H,column=index%H;float sum=0.f;
#pragma unroll
    for(int slot=0;slot<Top;++slot){size_t r=token*Top+slot;if(ids[r]>=0 && ids[r]<E){float product=routes[r*H+column]*weights[r];sum=sum+product;}}
    diagnostic[index]=sum;uint16_t result=bf16_rne(sum);out[index]=result;
    if(!isfinite(sum)||(result&0x7f80u)==0x7f80u)atomicOr(sticky,32u);
}
}
#include "routed_storage_n32_prefill.h"
extern "C" hipError_t ornith_routed_direct_get_layout(int Tcap,OrnithRoutedDirectLayout* out){
    using namespace routed_direct;if(!out||Tcap<0||Tcap>2048)return hipErrorInvalidValue;
    OrnithRoutedDirectLayout l{};size_t cursor=0,t=Tcap,r=t*Top;
    auto add=[&](size_t& field,size_t bytes){field=cursor;cursor=align256(cursor+bytes);};
    add(l.gate_x,t*H*4);add(l.gate_low,t*H/2);add(l.gate_high,t*H/2);
    add(l.gate_scales,t*(H/G)*2);add(l.gate_sums,t*(H/G)*4);add(l.gate_y,r*(2*I)*4);
    add(l.middle,r*I*4);add(l.down_x,r*I*4);add(l.down_low,r*I/2);add(l.down_high,r*I/2);
    add(l.down_scales,r*(I/G)*2);add(l.down_sums,r*(I/G)*4);add(l.route_out,r*H*4);add(l.final_f32,t*H*4);
    l.workspace_bytes=Tcap>8?prefill_layout(cursor,Tcap).bytes:cursor;*out=l;return hipSuccess;
}
template<bool A8> static hipError_t routed_direct_launch(const void* input,const void* weights,const void* ids,
        const void* gate_codes,const void* gate_meta,const void* down_codes,const void* down_meta,
        void* workspace,size_t bytes,void* output,void* flags,int T,int Tcap,hipStream_t stream){
    using namespace routed_direct;OrnithRoutedDirectLayout l{};
    if(ornith_routed_direct_get_layout(Tcap,&l)!=hipSuccess||T<0||T>Tcap||bytes<l.workspace_bytes)return hipErrorInvalidValue;
    if(!T)return hipSuccess;
    if(!input||!weights||!ids||!gate_codes||!gate_meta||!down_codes||!down_meta||!workspace||!output||!flags)return hipErrorInvalidValue;
    auto sticky=static_cast<uint32_t*>(flags);auto rid=static_cast<const int32_t*>(ids);int R=T*Top;
#define CHECK_LAUNCH() {auto s=hipGetLastError();if(s!=hipSuccess)return s;}
    PrefillLayout p{};
    if(T>16){
        p=prefill_layout(align256(l.final_f32+size_t(Tcap)*H*4),Tcap);
        auto s=hipMemsetAsync(at<int32_t>(workspace,p.counts),0,E*4,stream);if(s!=hipSuccess)return s;
        histogram<<<(R+255)/256,256,0,stream>>>(rid,at<int32_t>(workspace,p.counts),sticky,R);CHECK_LAUNCH();
        scan_routes<<<1,1,0,stream>>>(at<int32_t>(workspace,p.counts),at<int32_t>(workspace,p.offsets),at<int32_t>(workspace,p.cursors));CHECK_LAUNCH();
        scatter<<<(R+255)/256,256,0,stream>>>(rid,at<int32_t>(workspace,p.offsets),at<int32_t>(workspace,p.cursors),at<int32_t>(workspace,p.routes),R);CHECK_LAUNCH();
        partition<<<1,1,0,stream>>>(at<int32_t>(workspace,p.counts),at<int32_t>(workspace,p.offsets),at<int32_t>(workspace,p.routes),at<Descriptor>(workspace,p.descriptors),at<int32_t>(workspace,p.compact),at<uint32_t>(workspace,p.queue));CHECK_LAUNCH();
    }
    transform<true><<<T*(H/128),128,0,stream>>>(input,at<float>(workspace,l.gate_x),sticky,rid);CHECK_LAUNCH();
    quantize<A8><<<T*(H/G),32,0,stream>>>(at<float>(workspace,l.gate_x),at<uint32_t>(workspace,l.gate_low),at<uint32_t>(workspace,l.gate_high),at<__half>(workspace,l.gate_scales),at<int32_t>(workspace,l.gate_sums),sticky);CHECK_LAUNCH();
#define PROJECT(N,K,GATE,CODES,META,LO,HI,SCALES,SUMS,Y) \
    if(T<=16){projection_n32<N,K,GATE,A8><<<R*(N/32),32,0,stream>>>(rid,static_cast<const uint32_t*>(CODES),static_cast<const Meta*>(META),at<uint32_t>(workspace,l.LO),at<uint32_t>(workspace,l.HI),at<__half>(workspace,l.SCALES),at<int32_t>(workspace,l.SUMS),at<float>(workspace,l.Y),sticky);} \
    else{grouped_projection<N,K,GATE,A8><<<512,32,0,stream>>>(rid,static_cast<const uint32_t*>(CODES),static_cast<const Meta*>(META),at<uint32_t>(workspace,l.LO),at<uint32_t>(workspace,l.HI),at<__half>(workspace,l.SCALES),at<int32_t>(workspace,l.SUMS),at<float>(workspace,l.Y),sticky,at<Descriptor>(workspace,p.descriptors),at<int32_t>(workspace,p.routes),at<int32_t>(workspace,p.compact),at<uint32_t>(workspace,p.queue));}CHECK_LAUNCH();
    PROJECT(2*I,H,true,gate_codes,gate_meta,gate_low,gate_high,gate_scales,gate_sums,gate_y);
    middle_kernel<<<(R*I+255)/256,256,0,stream>>>(at<float>(workspace,l.gate_y),at<float>(workspace,l.middle),sticky,R*I,rid);CHECK_LAUNCH();
    transform<false><<<R*(I/128),128,0,stream>>>(at<float>(workspace,l.middle),at<float>(workspace,l.down_x),sticky,rid);CHECK_LAUNCH();
    quantize<A8><<<R*(I/G),32,0,stream>>>(at<float>(workspace,l.down_x),at<uint32_t>(workspace,l.down_low),at<uint32_t>(workspace,l.down_high),at<__half>(workspace,l.down_scales),at<int32_t>(workspace,l.down_sums),sticky);CHECK_LAUNCH();
    PROJECT(H,I,false,down_codes,down_meta,down_low,down_high,down_scales,down_sums,route_out);
    reduce_slots<<<(T*H+255)/256,256,0,stream>>>(at<float>(workspace,l.route_out),static_cast<const float*>(weights),rid,at<float>(workspace,l.final_f32),static_cast<uint16_t*>(output),sticky,T*H);CHECK_LAUNCH();
#undef CHECK_LAUNCH
#undef PROJECT
    return hipSuccess;
}
#define ROUTED_ENTRY(NAME,A8) \
extern "C" hipError_t NAME(const void* input,const void* weights,const void* ids, \
        const void* gate_codes,const void* gate_meta,const void* down_codes,const void* down_meta, \
        void* workspace,size_t bytes,void* output,void* flags,int T,int Tcap,hipStream_t stream){ \
    return routed_direct_launch<A8>(input,weights,ids,gate_codes,gate_meta,down_codes,down_meta, \
        workspace,bytes,output,flags,T,Tcap,stream); \
}
ROUTED_ENTRY(ornith_routed_direct_launch,true)
ROUTED_ENTRY(ornith_routed_direct_launch_a4,false)
#undef ROUTED_ENTRY
extern "C" hipError_t ornith_routed_direct_launch_a4_prefill(
        const void* input,const void* weights,const void* ids,
        const void* gate_codes,const void* gate_meta,const void* down_codes,const void* down_meta,
        void* workspace,size_t bytes,void* output,void* flags,int T,int Tcap,hipStream_t stream){
    if(T>64)return routed_direct_launch<false>(input,weights,ids,gate_codes,gate_meta,down_codes,down_meta,
        workspace,bytes,output,flags,T,Tcap,stream);
    return routed_direct_launch<true>(input,weights,ids,gate_codes,gate_meta,down_codes,down_meta,
        workspace,bytes,output,flags,T,Tcap,stream);
}

extern "C" ORNITH_ROUTED_DIRECT_EXPORT int ornith_routed_storage_n32(){return 32;}
