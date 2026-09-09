// Exact physical-layout derivative of frozen row-major4416c383...; arithmetic unchanged.
// Copyright2026 Ciru. Signed I8 full-head prototype; no routed IU4 changes.
// Control flow and exact quantization inherit Ciru's separately sealed dense ABI1.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <bit>
#include <cstdint>
#include <limits>
#include "head_i8_tile_api.h"
namespace head_i8_tile {
constexpr int K=2048,G=128,Groups=16;
constexpr float InvSqrt128=0x1.6a09e6p-4f;
static_assert(std::bit_cast<uint32_t>(InvSqrt128)==0x3db504f3u);
using i32x4=int32_t __attribute__((ext_vector_type(4)));
using i32x8=int32_t __attribute__((ext_vector_type(8)));
static_assert(sizeof(size_t)==8);
size_t align256(size_t n){return (n+255)&~size_t(255);}
// Physical words [N/16,K/16,4,N16], each uint32 packs four signed K bytes.
__device__ __forceinline__ size_t tile_word(int n,int kword){
    return ((size_t(n/16)*(K/16)+kword/4)*4+(kword&3))*16+(n&15);
}
template<class T>T* at(void* p,size_t offset){return reinterpret_cast<T*>(static_cast<uint8_t*>(p)+offset);}
bool shape(int N){return N==256||N==248320;}
__device__ __forceinline__ float divide(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ uint16_t bf16_rne(float value){
    uint32_t bits=__float_as_uint(value);
    if((bits&0x7fffffffu)>0x7f800000u)return uint16_t((bits>>16)|0x40u);
    return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
__global__ __launch_bounds__(128) void transform(const uint16_t* input,float* output,uint32_t* sticky){
    const size_t base=size_t(blockIdx.x)*128;const int lane=threadIdx.x;
    const float value=__uint_as_float(uint32_t(input[base+lane])<<16);
    if(!isfinite(value))atomicOr(sticky,unsigned(ORNITH_HEAD_I8_NONFINITE_INPUT_TRANSFORM));
    __shared__ float first[128],second[128];first[lane]=value;__syncthreads();
    float* src=first;float* dst=second;
#pragma unroll
    for(int step=1;step<128;step*=2){
        const int lo=lane&~step;
        dst[lane]=(lane&step)?src[lo]-src[lo+step]:src[lo]+src[lo+step];
        __syncthreads();float* old=src;src=dst;dst=old;
    }
    const float result=src[lane]*InvSqrt128;output[base+lane]=result;
    if(!isfinite(result))atomicOr(sticky,unsigned(ORNITH_HEAD_I8_NONFINITE_INPUT_TRANSFORM));
}
__global__ __launch_bounds__(32) void encode_a8(const float* transformed,uint32_t* words,
        __half* scales,uint32_t* sticky){
    const int lane=threadIdx.x;const size_t group=blockIdx.x;
    float values[4],maximum=0.f;
#pragma unroll
    for(int d=0;d<4;++d){
        const float value=transformed[group*128+lane*4+d];
        const bool finite=isfinite(value);
        if(!finite)atomicOr(sticky,unsigned(ORNITH_HEAD_I8_INVALID_ACTIVATION));
        values[d]=finite?value:0.f;maximum=fmaxf(maximum,fabsf(values[d]));
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)maximum=fmaxf(maximum,__shfl_xor(maximum,delta,32));
    const float raw=divide(maximum,127.f);
    const bool valid=isfinite(raw)&&raw<=65504.f;
    if(!valid)atomicOr(sticky,unsigned(ORNITH_HEAD_I8_INVALID_ACTIVATION));
    const __half scale=__float2half_rn(maximum==0.f||!valid?1.f:fmaxf(raw,0x1p-14f));
    uint32_t packed=0;
#pragma unroll
    for(int d=0;d<4;++d){
        int q=maximum==0.f||!valid?0:__float2int_rn(divide(values[d],__half2float(scale)));
        q=q< -127?-127:(q>127?127:q);
        packed|=(uint32_t(q)&255u)<<(d*8);
    }
    words[group*32+lane]=packed;
    if(lane==0)scales[group]=scale;
}
__device__ __forceinline__ void store(float value,size_t index,float* diagnostic,
        uint16_t* output,uint32_t* sticky){
    if(!isfinite(value))atomicOr(sticky,unsigned(ORNITH_HEAD_I8_NONFINITE_PROJECTION));
    const uint16_t bf=bf16_rne(value);
    if((bf&0x7f80u)==0x7f80u)atomicOr(sticky,unsigned(ORNITH_HEAD_I8_NONFINITE_BF16));
    diagnostic[index]=value;output[index]=bf;
}
template<int N,bool Diagnostic>__global__ __launch_bounds__(32) void compact_projection(
        const uint32_t* weights,const __half* weight_scales,const uint32_t* activations,
        const __half* activation_scales,float* diagnostic,uint16_t* output,
        int32_t* group_dots,uint32_t* sticky){
    constexpr int NTiles=N/8;
    const int lane=threadIdx.x,piece=lane&3;
    const int row=blockIdx.x/NTiles,n=(blockIdx.x%NTiles)*8+lane/4;
    const size_t ab=size_t(row)*(K/4);
    float value=0.f;
    for(int group=0;group<Groups;++group){
        int dot=0;
#pragma unroll
        for(int word=0;word<8;++word){
            const int kword=group*32+piece*8+word;
            dot=__builtin_amdgcn_sudot4(true,int(weights[tile_word(n,kword)]),
                true,int(activations[ab+kword]),dot,false);
        }
        dot=dot+__shfl_xor(dot,1,32);dot=dot+__shfl_xor(dot,2,32);
        if(piece==0){
            if constexpr(Diagnostic)group_dots[(size_t(row)*N+n)*Groups+group]=dot;
            const float sa=__half2float(activation_scales[size_t(row)*Groups+group]);
            const float sw=__half2float(weight_scales[size_t(n)*Groups+group]);
            const float scaled=sa*float(dot),term=sw*scaled;
            value=value+term;
        }
    }
    if(piece==0)store(value,size_t(row)*N+n,diagnostic,output,sticky);
}
template<int N,bool Diagnostic>__global__ __launch_bounds__(32) void wmma_projection(
        const uint32_t* weights,const __half* weight_scales,const uint32_t* activations,
        const __half* activation_scales,float* diagnostic,uint16_t* output,
        int32_t* group_dots,uint32_t* sticky,int M){
    constexpr int NTiles=N/16;
    const int lane=threadIdx.x,lane16=lane&15,half=lane>>4;
    const int row=(blockIdx.x/NTiles)*16+lane16,nbase=(blockIdx.x%NTiles)*16;
    const bool live=row<M;
    const size_t ab=size_t(row)*(K/4);
    float values[8]={};
    for(int group=0;group<Groups;++group){
        i32x8 dots={};
#pragma unroll
        for(int step=0;step<8;++step){
            i32x4 w,a;const int first=group*32+step*4;
#pragma unroll
            for(int d=0;d<4;++d){w[d]=weights[tile_word(nbase+lane16,first+d)];a[d]=live?activations[ab+first+d]:0;}
            dots=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,w,true,a,dots,false);
        }
        if(live){
            const float sa=__half2float(activation_scales[size_t(row)*Groups+group]);
#pragma unroll
            for(int v=0;v<8;++v){
                const int n=nbase+2*v+half,dot=dots[v];
                if constexpr(Diagnostic)group_dots[(size_t(row)*N+n)*Groups+group]=dot;
                const float sw=__half2float(weight_scales[size_t(n)*Groups+group]);
                const float scaled=sa*float(dot),term=sw*scaled;
                values[v]=values[v]+term;
            }
        }
    }
    if(live){
#pragma unroll
        for(int v=0;v<8;++v)store(values[v],size_t(row)*N+nbase+2*v+half,diagnostic,output,sticky);
    }
}
template<int N,bool Diagnostic>__global__ __launch_bounds__(128) void shared64_projection(
        const uint32_t* weights,const __half* weight_scales,const uint32_t* activations,
        const __half* activation_scales,float* diagnostic,uint16_t* output,
        int32_t* group_dots,uint32_t* sticky,int M){
    constexpr int NTiles=N/16;
    __shared__ uint32_t shared_weights[512];
    const int lane=threadIdx.x&31,wave=threadIdx.x>>5,lane16=lane&15,half=lane>>4;
    const int row=(blockIdx.x/NTiles)*64+wave*16+lane16,nbase=(blockIdx.x%NTiles)*16;
    const bool live=row<M;
    const size_t ab=size_t(row)*(K/4);
    float values[8]={};
    for(int group=0;group<Groups;++group){
        for(int i=threadIdx.x;i<512;i+=128)
            shared_weights[i]=weights[tile_word(nbase+i%16,group*32+i/16)];
        __syncthreads();
        i32x8 dots={};
#pragma unroll
        for(int step=0;step<8;++step){
            i32x4 w,a;const int first=group*32+step*4;
#pragma unroll
            for(int d=0;d<4;++d){w[d]=shared_weights[(step*4+d)*16+lane16];a[d]=live?activations[ab+first+d]:0;}
            dots=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,w,true,a,dots,false);
        }
        if(live){
            const float sa=__half2float(activation_scales[size_t(row)*Groups+group]);
#pragma unroll
            for(int v=0;v<8;++v){
                const int n=nbase+2*v+half,dot=dots[v];
                if constexpr(Diagnostic)group_dots[(size_t(row)*N+n)*Groups+group]=dot;
                const float sw=__half2float(weight_scales[size_t(n)*Groups+group]);
                const float scaled=sa*float(dot),term=sw*scaled;
                values[v]=values[v]+term;
            }
        }
        __syncthreads();
    }
    if(live){
#pragma unroll
        for(int v=0;v<8;++v)store(values[v],size_t(row)*N+nbase+2*v+half,diagnostic,output,sticky);
    }
}
template<int N,bool Diagnostic>hipError_t dispatch(const void* weights,const void* scales,
        void* workspace,void* output,void* group_dots,void* flags,int M,int geometry,
        OrnithHeadI8Layout layout,hipStream_t stream){
    const auto w=static_cast<const uint32_t*>(weights);
    const auto sw=static_cast<const __half*>(scales);
    const auto a=at<uint32_t>(workspace,layout.activation_words);
    const auto sa=at<__half>(workspace,layout.activation_scales);
    const auto diagnostic=at<float>(workspace,layout.projection);
    const auto out=static_cast<uint16_t*>(output);
    const auto dots=static_cast<int32_t*>(group_dots);
    const auto sticky=static_cast<uint32_t*>(flags);
    const bool compact=geometry==1||(geometry==2&&M<=4);
    if(compact)compact_projection<N,Diagnostic><<<M*(N/8),32,0,stream>>>(w,sw,a,sa,diagnostic,out,dots,sticky);
    else if(M>=32)shared64_projection<N,Diagnostic><<<((M+63)/64)*(N/16),128,0,stream>>>(w,sw,a,sa,diagnostic,out,dots,sticky,M);
    else wmma_projection<N,Diagnostic><<<((M+15)/16)*(N/16),32,0,stream>>>(w,sw,a,sa,diagnostic,out,dots,sticky,M);
    return hipGetLastError();
}
struct Range{uintptr_t lo,hi;};
bool range(const void* pointer,size_t bytes,Range& out){
    const uintptr_t lo=reinterpret_cast<uintptr_t>(pointer);
    if(!pointer||bytes>std::numeric_limits<uintptr_t>::max()-lo)return false;
    out={lo,lo+bytes};return true;
}
}
extern "C" uint32_t ornith_head_i8_tile_abi_version(){return 1;}
extern "C" hipError_t ornith_head_i8_tile_get_layout(int Mcap,int N,OrnithHeadI8Layout* out){
    using namespace head_i8_tile;
    if(!out||Mcap<0||Mcap>64||!shape(N))return hipErrorInvalidValue;
    OrnithHeadI8Layout layout{};size_t cursor=0,m=Mcap;
    auto add=[&](size_t& offset,size_t bytes){offset=cursor;cursor=align256(cursor+bytes);};
    add(layout.transformed,m*K*4);add(layout.activation_words,m*K);
    add(layout.activation_scales,m*Groups*2);add(layout.projection,m*N*4);
    layout.workspace_bytes=cursor;*out=layout;return hipSuccess;
}
extern "C" hipError_t ornith_head_i8_tile_launch(const void* input,const void* weights,const void* scales,
        void* workspace,size_t workspace_bytes,void* output,void* group_dots,size_t group_dots_bytes,
        void* flags,int M,int Mcap,int N,int geometry,hipStream_t stream){
    using namespace head_i8_tile;OrnithHeadI8Layout layout{};
    if(ornith_head_i8_tile_get_layout(Mcap,N,&layout)!=hipSuccess||M<0||M>Mcap||geometry<0||geometry>2)
        return hipErrorInvalidValue;
    if((group_dots==nullptr)!=(group_dots_bytes==0)
            ||(group_dots&&group_dots_bytes<size_t(M)*N*Groups*4))return hipErrorInvalidValue;
    if(M==0)return hipSuccess;
    if(workspace_bytes<layout.workspace_bytes)return hipErrorInvalidValue;
    const void* pointers[]={input,weights,scales,workspace,output,group_dots,flags};
    const size_t sizes[]={size_t(M)*K*2,size_t(N)*K,size_t(N)*Groups*2,
        workspace_bytes,size_t(M)*N*2,group_dots_bytes,4};
    const size_t alignments[]={2,4,2,256,2,4,4};Range ranges[7]{};
    for(int i=0;i<7;++i){
        if(i==5&&!group_dots)continue;
        if(reinterpret_cast<uintptr_t>(pointers[i])%alignments[i]||!range(pointers[i],sizes[i],ranges[i]))
            return hipErrorInvalidValue;
    }
    for(int i=3;i<7;++i)for(int j=0;j<7;++j){
        if(i==j||(!group_dots&&(i==5||j==5)))continue;
        if(ranges[i].lo<ranges[j].hi&&ranges[j].lo<ranges[i].hi)return hipErrorInvalidValue;
    }
    auto tx=at<float>(workspace,layout.transformed);
    auto sticky=static_cast<uint32_t*>(flags);
    transform<<<M*Groups,128,0,stream>>>(static_cast<const uint16_t*>(input),tx,sticky);
    auto status=hipGetLastError();if(status!=hipSuccess)return status;
    encode_a8<<<M*Groups,32,0,stream>>>(tx,at<uint32_t>(workspace,layout.activation_words),
        at<__half>(workspace,layout.activation_scales),sticky);
    status=hipGetLastError();if(status!=hipSuccess)return status;
#define HEAD_DISPATCH(NV) if(N==NV){if(group_dots)return dispatch<NV,true>(weights,scales,workspace,output,group_dots,flags,M,geometry,layout,stream);return dispatch<NV,false>(weights,scales,workspace,output,nullptr,flags,M,geometry,layout,stream);}
    HEAD_DISPATCH(256) HEAD_DISPATCH(248320)
#undef HEAD_DISPATCH
    return hipErrorInvalidValue;
}
