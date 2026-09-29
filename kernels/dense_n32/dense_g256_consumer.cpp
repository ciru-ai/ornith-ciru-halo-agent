// Generic dense G256 extension of the measured QKVZ arithmetic and N16 geometry.
// Copyright (c) Ciru. Direct dense IU4 projection; arithmetic derived from
// Ciru grouped ABI2, with no routed-expert storage or routing work.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <bit>
#include <cstdint>
#include <limits>
#include "dense_g256_api.h"
namespace dense_g256 {
constexpr int G=256,Words=G/8,Parts=G/32;
constexpr float InvSqrt128=0x1.6a09e6p-4f;
static_assert(std::bit_cast<uint32_t>(InvSqrt128)==0x3db504f3u);
using i32x2=int32_t __attribute__((ext_vector_type(2)));
using i32x8=int32_t __attribute__((ext_vector_type(8)));
struct Meta { __half scale,offset; };
static_assert(sizeof(Meta)==4 && sizeof(size_t)==8);
size_t align256(size_t n){return (n+255)&~size_t(255);}
template<class T>T* at(void* w,size_t offset){return offset==SIZE_MAX?nullptr:reinterpret_cast<T*>(static_cast<uint8_t*>(w)+offset);}
bool shape(int N,int K){return N>0 && N<=12288 && N%16==0 && K>0 && K<=8192 && K%G==0;}
__device__ __forceinline__ float divide(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ uint16_t bf16_rne(float value){
    unsigned bits=__float_as_uint(value);
    if((bits&0x7fffffffu)>0x7f800000u)return uint16_t((bits>>16)|0x40u);
    return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
__global__ __launch_bounds__(128) void transform(const uint16_t* x,float* y,uint32_t* sticky){
    size_t base=size_t(blockIdx.x)*128;int lane=threadIdx.x;
    float value=__uint_as_float(uint32_t(x[base+lane])<<16);
    if(!isfinite(value))atomicOr(sticky,unsigned(ORNITH_DENSE_G256_NONFINITE_INPUT_TRANSFORM));
    __shared__ float first[128],second[128];first[lane]=value;__syncthreads();
    float* src=first;float* dst=second;
#pragma unroll
    for(int step=1;step<128;step*=2){
        int lo=lane&~step;dst[lane]=(lane&step)?src[lo]-src[lo+step]:src[lo]+src[lo+step];
        __syncthreads();float* swap=src;src=dst;dst=swap;
    }
    float result=src[lane]*InvSqrt128;y[base+lane]=result;
    if(!isfinite(result))atomicOr(sticky,unsigned(ORNITH_DENSE_G256_NONFINITE_INPUT_TRANSFORM));
}
template<bool A8>__global__ __launch_bounds__(32) void quantize(const float* x,uint32_t* low,uint32_t* high,
        __half* scales,int32_t* sums,uint32_t* sticky){
    int lane=threadIdx.x;size_t group=blockIdx.x;
    float values[Parts],absmax=0.f;
#pragma unroll
    for(int part=0;part<Parts;++part){
        float original=x[group*G+part*32+lane];bool finite=isfinite(original);
        if(!finite)atomicOr(sticky,unsigned(ORNITH_DENSE_G256_INVALID_ACTIVATION));
        values[part]=finite?original:0.f;absmax=fmaxf(absmax,fabsf(values[part]));
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)absmax=fmaxf(absmax,__shfl_xor(absmax,delta,32));
    constexpr int qmax=A8?127:7;
    float raw=divide(absmax,float(qmax));bool valid=isfinite(raw)&&raw<=65504.f;
    if(!valid)atomicOr(sticky,unsigned(ORNITH_DENSE_G256_INVALID_ACTIVATION));
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
__device__ __forceinline__ void store(float value,size_t index,float* diagnostic,uint16_t* out,uint32_t* sticky){
    if(!isfinite(value))atomicOr(sticky,unsigned(ORNITH_DENSE_G256_NONFINITE_PROJECTION));
    uint16_t result=bf16_rne(value);
    if((result&0x7f80u)==0x7f80u)atomicOr(sticky,unsigned(ORNITH_DENSE_G256_NONFINITE_BF16));
    diagnostic[index]=value;out[index]=result;
}
template<bool A8>__global__ __launch_bounds__(32) void wmma_projection(
        const uint32_t* weights,const Meta* metadata,const uint32_t* low,const uint32_t* high,
        const __half* scales,const int32_t* sums,float* diagnostic,uint16_t* out,uint32_t* sticky,int M,int N,int K){
    const int Groups=K/G,NTiles=N/16;
    int lane=threadIdx.x,lane16=lane&15,half=lane>>4;
    int row=(blockIdx.x/NTiles)*16+lane16,nbase=(blockIdx.x%NTiles)*16;bool live=row<M;
    size_t abase=size_t(row)*(K/8);float values[8]={};
    for(int group=0;group<Groups;++group){
        i32x8 dl={},dh={};
#pragma unroll
        for(int step=0;step<Words/2;++step){
            int word=group*Words+step*2;i32x2 w,lo,hi;
            w[0]=weights[((size_t(nbase/16)*Groups+group)*Words+step*2)*16+lane16];w[1]=weights[((size_t(nbase/16)*Groups+group)*Words+step*2+1)*16+lane16];
            lo[0]=live?low[abase+word]:0;lo[1]=live?low[abase+word+1]:0;
            if constexpr(A8){
                hi[0]=live?high[abase+word]:0;hi[1]=live?high[abase+word+1]:0;
                dl=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false,w,false,lo,dl,false);
                dh=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false,w,true,hi,dh,false);
            }else dl=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false,w,true,lo,dl,false);
        }
        if(live){
            float sa=__half2float(scales[size_t(row)*Groups+group]);
            float scaled_sum=sa*float(sums[size_t(row)*Groups+group]);
#pragma unroll
            for(int v=0;v<8;++v){
                int n=nbase+2*v+half,dot=A8?dl[v]+16*dh[v]:dl[v];Meta wm=metadata[(size_t(n/16)*Groups+group)*16+(n&15)];
                float scaled_dot=sa*float(dot),product=__half2float(wm.scale)*scaled_dot;
                float correction=__half2float(wm.offset)*scaled_sum;values[v]=values[v]+(product+correction);
            }
        }
    }
    if(live){
#pragma unroll
        for(int v=0;v<8;++v)store(values[v],size_t(row)*N+nbase+2*v+half,diagnostic,out,sticky);
    }
}
template<bool A8>__global__ __launch_bounds__(32) void compact_projection(
        const uint32_t* weights,const Meta* metadata,const uint32_t* low,const uint32_t* high,
        const __half* scales,const int32_t* sums,float* diagnostic,uint16_t* out,uint32_t* sticky,int N,int K){
    const int Groups=K/G,NTiles=N/16;
    int lane=threadIdx.x,piece=lane>>4,row=blockIdx.x/NTiles,n=(blockIdx.x%NTiles)*16+(lane&15);
    size_t abase=size_t(row)*(K/8);float value=0.f;
    for(int group=0;group<Groups;++group){
        int dot=0;
#pragma unroll
        for(int j=0;j<Words/2;++j){
            int local=piece*(Words/2)+j,word=group*Words+local;
            int w=int(weights[((size_t(n/16)*Groups+group)*Words+local)*16+(n&15)]);
            int lo=int(low[abase+word]),hi=int(high[abase+word]);
            int dl=__builtin_amdgcn_sudot8(false,w,false,lo,0,false);
            int dh=__builtin_amdgcn_sudot8(false,w,true,hi,0,false);
            dot+=dl+16*dh;
        }
        dot=dot+__shfl_xor(dot,16,32);
        if(piece==0){
            float sa=__half2float(scales[size_t(row)*Groups+group]);
            Meta wm=metadata[(size_t(n/16)*Groups+group)*16+(n&15)];
            float scaled_dot=sa*float(dot),scaled_sum=sa*float(sums[size_t(row)*Groups+group]);
            float product=__half2float(wm.scale)*scaled_dot,correction=__half2float(wm.offset)*scaled_sum;
            value=value+(product+correction);
        }
    }
    if(piece==0)store(value,size_t(row)*N+n,diagnostic,out,sticky);
}
template<bool A8>hipError_t dispatch(const void* codes,const void* meta,void* workspace,
        void* output,void* flags,int M,int N,int K,int geometry,OrnithDenseG256Layout l,hipStream_t stream){
    bool compact=geometry==1||(geometry==2&&M<=4);
    auto w=static_cast<const uint32_t*>(codes);auto wm=static_cast<const Meta*>(meta);
    auto lo=at<uint32_t>(workspace,l.low),hi=at<uint32_t>(workspace,l.high);
    auto sc=at<__half>(workspace,l.scales);auto sums=at<int32_t>(workspace,l.sums);
    auto diag=at<float>(workspace,l.projection);auto out=static_cast<uint16_t*>(output);auto sticky=static_cast<uint32_t*>(flags);
    if(compact)compact_projection<A8><<<M*(N/16),32,0,stream>>>(w,wm,lo,hi,sc,sums,diag,out,sticky,N,K);
    else wmma_projection<A8><<<((M+15)/16)*(N/16),32,0,stream>>>(w,wm,lo,hi,sc,sums,diag,out,sticky,M,N,K);
    return hipGetLastError();
}
struct Range{uintptr_t lo,hi;};
bool range(const void* p,size_t n,Range& out){
    uintptr_t lo=reinterpret_cast<uintptr_t>(p);
    if(!p||n>std::numeric_limits<uintptr_t>::max()-lo)return false;out={lo,lo+n};return true;
}
}
extern "C" uint32_t ornith_dense_g256_abi_version(){return 1;}
extern "C" hipError_t ornith_dense_g256_get_layout(int Mcap,int N,int K,int bits,OrnithDenseG256Layout* out){
    using namespace dense_g256;
    if(!out||Mcap<0||Mcap>8192||!shape(N,K)||bits!=8)return hipErrorInvalidValue;
    OrnithDenseG256Layout l{};size_t cursor=0,m=Mcap;
    auto add=[&](size_t& offset,size_t bytes){offset=cursor;cursor=align256(cursor+bytes);};
    add(l.transformed,m*K*4);add(l.low,m*K/2);
    add(l.high,m*K/2);
    add(l.scales,m*(K/G)*2);add(l.sums,m*(K/G)*4);add(l.projection,m*N*4);
    l.workspace_bytes=cursor;*out=l;return hipSuccess;
}
extern "C" hipError_t ornith_dense_g256_launch(const void* input,const void* codes,const void* metadata,
        void* workspace,size_t bytes,void* output,void* flags,int M,int Mcap,int N,int K,int bits,int transform_kind,int geometry,hipStream_t stream){
    using namespace dense_g256;OrnithDenseG256Layout l{};
    if(ornith_dense_g256_get_layout(Mcap,N,K,bits,&l)!=hipSuccess||M<0||M>Mcap||transform_kind!=128||geometry<0||geometry>2)return hipErrorInvalidValue;
    if(M==0)return hipSuccess;
    const void* ptr[]={input,codes,metadata,workspace,output,flags};
    const size_t sizes[]={size_t(M)*K*2,size_t(N)*K/2,size_t(N)*(K/G)*4,bytes,size_t(M)*N*2,4};
    const size_t alignment[]={2,4,4,256,2,4};Range ranges[6];
    if(bytes<l.workspace_bytes)return hipErrorInvalidValue;
    for(int i=0;i<6;++i)if(reinterpret_cast<uintptr_t>(ptr[i])%alignment[i]||!range(ptr[i],sizes[i],ranges[i]))return hipErrorInvalidValue;
    for(int i=3;i<6;++i)for(int j=0;j<6;++j)if(i!=j&&ranges[i].lo<ranges[j].hi&&ranges[j].lo<ranges[i].hi)return hipErrorInvalidValue;
    auto sticky=static_cast<uint32_t*>(flags);auto tx=at<float>(workspace,l.transformed);
    transform<<<M*(K/128),128,0,stream>>>(static_cast<const uint16_t*>(input),tx,sticky);
    auto status=hipGetLastError();if(status!=hipSuccess)return status;
    quantize<true><<<M*(K/G),32,0,stream>>>(tx,at<uint32_t>(workspace,l.low),at<uint32_t>(workspace,l.high),at<__half>(workspace,l.scales),at<int32_t>(workspace,l.sums),sticky);
    status=hipGetLastError();if(status!=hipSuccess)return status;
    return dispatch<true>(codes,metadata,workspace,output,flags,M,N,K,geometry,l,stream);
}

namespace dense_g256 {
__global__ __launch_bounds__(128) void transform_bf16(const uint16_t* x,uint16_t* y,uint32_t* sticky){
    size_t base=size_t(blockIdx.x)*128;int lane=threadIdx.x;
    float value=__uint_as_float(uint32_t(x[base+lane])<<16);
    if(!isfinite(value))atomicOr(sticky,unsigned(ORNITH_DENSE_G256_NONFINITE_INPUT_TRANSFORM));
    __shared__ float first[128],second[128];first[lane]=value;__syncthreads();
    float* src=first;float* dst=second;
#pragma unroll
    for(int step=1;step<128;step*=2){
        int lo=lane&~step;dst[lane]=(lane&step)?src[lo]-src[lo+step]:src[lo]+src[lo+step];
        __syncthreads();float* swap=src;src=dst;dst=swap;
    }
    float result=src[lane]*InvSqrt128;uint16_t rounded=bf16_rne(result);y[base+lane]=rounded;
    if(!isfinite(result))atomicOr(sticky,unsigned(ORNITH_DENSE_G256_NONFINITE_INPUT_TRANSFORM));
    if((rounded&0x7f80u)==0x7f80u)atomicOr(sticky,unsigned(ORNITH_DENSE_G256_NONFINITE_BF16));
}
// One block reads a contiguous physical N16/K128 tile and transposes through
// shared memory so that the unpacked BF16[N,K] writes are contiguous as well.
__global__ __launch_bounds__(256) void dequantize_bf16(const uint32_t* codes,const Meta* metadata,
        uint16_t* out,uint32_t* sticky,int K){
    int lane=threadIdx.x,n=lane&15,word=lane>>4,groups=K/G;
    int ntile=blockIdx.x/(K/128),chunk=blockIdx.x%(K/128),group=chunk/2,half=chunk&1;
    uint32_t packed=codes[((size_t(ntile)*groups+group)*Words+half*16+word)*16+n];
    Meta meta=metadata[(size_t(ntile)*groups+group)*16+n];
    float scale=__half2float(meta.scale),offset=__half2float(meta.offset);
    __shared__ uint16_t tile[16*128];
#pragma unroll
    for(int i=0;i<8;++i){
        float value=scale*float((packed>>(4*i))&15u)+offset;uint16_t rounded=bf16_rne(value);
        tile[n*128+word*8+i]=rounded;
        if((rounded&0x7f80u)==0x7f80u)atomicOr(sticky,unsigned(ORNITH_DENSE_G256_NONFINITE_BF16));
    }
    __syncthreads();
#pragma unroll
    for(int i=0;i<8;++i){
        int linear=lane*8+i;
        out[size_t(ntile*16+linear/128)*K+chunk*128+linear%128]=tile[linear];
    }
}
}
extern "C" hipError_t ornith_dense_g256_prepare_bf16(const void* input,const void* codes,const void* metadata,
        void* transformed,void* weights,void* flags,int M,int N,int K,hipStream_t stream){
    using namespace dense_g256;
    if(M<0||M>8192||!shape(N,K))return hipErrorInvalidValue;
    if(M==0)return hipSuccess;
    const void* ptr[]={input,codes,metadata,transformed,weights,flags};
    const size_t sizes[]={size_t(M)*K*2,size_t(N)*K/2,size_t(N)*(K/G)*4,size_t(M)*K*2,size_t(N)*K*2,4};
    const size_t alignment[]={2,4,4,2,2,4};Range ranges[6];
    for(int i=0;i<6;++i)if(reinterpret_cast<uintptr_t>(ptr[i])%alignment[i]||!range(ptr[i],sizes[i],ranges[i]))return hipErrorInvalidValue;
    for(int i=3;i<6;++i)for(int j=0;j<6;++j)if(i!=j&&ranges[i].lo<ranges[j].hi&&ranges[j].lo<ranges[i].hi)return hipErrorInvalidValue;
    auto sticky=static_cast<uint32_t*>(flags);
    transform_bf16<<<M*(K/128),128,0,stream>>>(static_cast<const uint16_t*>(input),static_cast<uint16_t*>(transformed),sticky);
    auto status=hipGetLastError();if(status!=hipSuccess)return status;
    dequantize_bf16<<<(N/16)*(K/128),256,0,stream>>>(static_cast<const uint32_t*>(codes),static_cast<const Meta*>(metadata),static_cast<uint16_t*>(weights),sticky,K);
    return hipGetLastError();
}

// Isolated M1 N32 derivative; original N16 entrypoint and WMMA unchanged.
namespace dense_g256 {
__global__ __launch_bounds__(32) void compact_projection_n32(
        const uint32_t* weights,const Meta* metadata,const uint32_t* low,const uint32_t* high,
        const __half* scales,const int32_t* sums,float* diagnostic,uint16_t* out,uint32_t* sticky,int N,int K){
    const int Groups=K/G;int lane=threadIdx.x,n=blockIdx.x*32+lane;
    float value=0.f;
    for(int group=0;group<Groups;++group){
        int dot=0;
#pragma unroll
        for(int j=0;j<Words;++j){
            int word=group*Words+j;
            int w=int(weights[((size_t(n/32)*Groups+group)*Words+j)*32+(n&31)]);
            int lo=int(low[word]),hi=int(high[word]);
            int dl=__builtin_amdgcn_sudot8(false,w,false,lo,0,false);
            int dh=__builtin_amdgcn_sudot8(false,w,true,hi,0,false);
            dot+=dl+16*dh;
        }
        float sa=__half2float(scales[group]);
        Meta wm=metadata[(size_t(n/32)*Groups+group)*32+(n&31)];
        float scaled_dot=sa*float(dot),scaled_sum=sa*float(sums[group]);
        float product=__half2float(wm.scale)*scaled_dot,correction=__half2float(wm.offset)*scaled_sum;
        value=value+(product+correction);
    }
    store(value,n,diagnostic,out,sticky);
}
}
extern "C" __attribute__((visibility("default"))) hipError_t ornith_dense_g256_launch_n32(
        const void* input,const void* codes,const void* metadata,void* workspace,size_t bytes,
        void* output,void* flags,int M,int Mcap,int N,int K,int bits,int transform_kind,int geometry,hipStream_t stream){
    using namespace dense_g256;OrnithDenseG256Layout l{};
    if(M!=1||N%32||transform_kind!=128||geometry!=2||ornith_dense_g256_get_layout(Mcap,N,K,bits,&l)!=hipSuccess||bytes<l.workspace_bytes)return hipErrorInvalidValue;
    auto sticky=static_cast<uint32_t*>(flags);auto tx=at<float>(workspace,l.transformed);
    transform<<<K/128,128,0,stream>>>(static_cast<const uint16_t*>(input),tx,sticky);
    auto status=hipGetLastError();if(status!=hipSuccess)return status;
    quantize<true><<<K/G,32,0,stream>>>(tx,at<uint32_t>(workspace,l.low),at<uint32_t>(workspace,l.high),at<__half>(workspace,l.scales),at<int32_t>(workspace,l.sums),sticky);
    status=hipGetLastError();if(status!=hipSuccess)return status;
    compact_projection_n32<<<N/32,32,0,stream>>>(static_cast<const uint32_t*>(codes),static_cast<const Meta*>(metadata),at<uint32_t>(workspace,l.low),at<uint32_t>(workspace,l.high),at<__half>(workspace,l.scales),at<int32_t>(workspace,l.sums),at<float>(workspace,l.projection),static_cast<uint16_t*>(output),sticky,N,K);
    return hipGetLastError();
}
