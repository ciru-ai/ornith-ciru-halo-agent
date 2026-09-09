// Copyright 2026 Ciru. Fixed-shape exploratory signed IU4 attention.
// Q1120/K262144/H16/KV2/D256, causal final1120 queries. No runtime integration.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>
using i32x2=int32_t __attribute__((ext_vector_type(2)));
using i32x8=int32_t __attribute__((ext_vector_type(8)));
using bf16=hip_bfloat16;
constexpr int Q=1120,D=256;
__device__ __forceinline__ float scale_for(float a){
    return a==0 ? 1.f : __half2float(__float2half_rn(fmaxf(a/7.f,0x1p-14f)));
}
__device__ __forceinline__ uint32_t code(float x,float s){
    int v=int(rintf(x/s)); v=v< -7? -7:(v>7?7:v);return uint32_t(v)&15;
}
template<bool IsQ, int KB> __global__ __launch_bounds__(256) void prep_qk(
        const bf16* src,uint32_t* packed,__half* scales,int tokens,int stride0,int stride1){
    constexpr int Heads=IsQ?16:2;int Tokens=IsQ?Q:tokens;
    int lane=threadIdx.x&31,row=blockIdx.x*8+(threadIdx.x>>5);
    if(row>=Heads*Tokens)return;
    int t=row/Heads,h=row%Heads;float x[8];
#pragma unroll
    for(int j=0;j<8;++j)x[j]=float(src[IsQ?(size_t(t)*stride0+h*stride1+lane*8+j):(size_t(row)*256+lane*8+j)]);
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
    else {packed[((size_t(h)*KB+t/32)*32+lane)*32+t%32]=word;if(lane==0)scales[(h*KB+t/32)*32+t%32]=__float2half_rn(s);}
}
template<int KB> __global__ __launch_bounds__(256) void prep_v(const bf16* src,uint32_t* packed,__half* scales){
    int block=blockIdx.x,h=blockIdx.y,col=threadIdx.x;float x[32];
#pragma unroll
    for(int j=0;j<32;++j)x[j]=float(src[(size_t(block*32+j)*2+h)*256+col]);
#pragma unroll
    for(int stride=1;stride<32;stride*=2){
#pragma unroll
        for(int j=0;j<32;++j)if((j&stride)==0){float a=x[j],b=x[j+stride];x[j]=a+b;x[j+stride]=a-b;}
    }
    float a=0;
#pragma unroll
    for(int j=0;j<32;++j){x[j]*=0.1767766952966369f;a=fmaxf(a,fabsf(x[j]));}
    float s=scale_for(a);scales[(h*KB+block)*256+col]=__float2half_rn(s);
#pragma unroll
    for(int w=0;w<4;++w){uint32_t word=0;
#pragma unroll
        for(int j=0;j<8;++j)word|=code(x[w*8+j],s)<<(4*j);
        packed[((size_t(h)*KB+block)*4+w)*256+col]=word;
    }
}
template<int KB> __global__ __launch_bounds__(256) void attention(const uint32_t* qp,const __half* qs,
        const uint32_t* kp,const __half* ks,const uint32_t* vp,const __half* vs,bf16* out,int tokens){
    __shared__ uint32_t sk[32*32],sv[4*256];
    __shared__ float sks[32],svs[256];
    int tid=threadIdx.x,lane=tid&31,row16=lane&15,piece=lane>>4,wave=tid>>5;
    int h=blockIdx.y,kh=h/8,row=blockIdx.x*128+wave*16+row16;
    bool live=row<Q;float sq=live?__half2float(qs[h*Q+row]):1.f;
    uint32_t q[32];
#pragma unroll
    for(int j=0;j<32;++j)q[j]=live?qp[(size_t(h)*Q+row)*32+j]:0;
    float accum[128]={},m=-INFINITY,l=0;
    // All lanes participate in barriers, including the final CTA query tail.
    for(int block=0;block<tokens/32;++block){
#pragma unroll
        for(int j=0;j<4;++j){int i=tid+256*j;sk[i]=kp[(size_t(kh)*KB+block)*1024+i];sv[i]=vp[(size_t(kh)*KB+block)*1024+i];}
        if(tid<32)sks[tid]=__half2float(ks[(kh*KB+block)*32+tid]);
        svs[tid]=__half2float(vs[(kh*KB+block)*256+tid]);
        __syncthreads();
        float p[16];
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
                p[tile*8+j]=(block*32+key<=tokens-Q+row)?score:-INFINITY;
            }
        }
        float mx=-INFINITY;
#pragma unroll
        for(int j=0;j<16;++j)mx=fmaxf(mx,p[j]);
        mx=fmaxf(mx,__shfl_xor(mx,16,32));
        float next=fmaxf(m,mx),alpha=exp2f((m-next)*1.4426950408889634f),sum=0;
#pragma unroll
        for(int j=0;j<16;++j){p[j]=exp2f((p[j]-next)*1.4426950408889634f);sum+=p[j];}
        sum+=__shfl_xor(sum,16,32);l=l*alpha+sum;m=next;
#pragma unroll
        for(int j=0;j<128;++j)accum[j]*=alpha;
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
        for(int j=0;j<16;++j){p[j]*=0.1767766952966369f;amax=fmaxf(amax,fabsf(p[j]));}
        amax=fmaxf(amax,__shfl_xor(amax,16,32));float sp=scale_for(amax);
        uint32_t pp[4]={};
#pragma unroll
        for(int w=0;w<4;++w){
#pragma unroll
            for(int j=0;j<4;++j){uint32_t own=code(p[w*4+j],sp),peer=__shfl_xor(own,16,32);
                pp[w]|=(piece?peer:own)<<(8*j);pp[w]|=(piece?own:peer)<<(8*j+4);
            }
        }
#pragma unroll
        for(int tile=0;tile<16;++tile){i32x8 dots={};
#pragma unroll
            for(int step=0;step<2;++step){
                i32x2 a={int32_t(sv[(2*step)*256+tile*16+row16]),int32_t(sv[(2*step+1)*256+tile*16+row16])};
                i32x2 b={int32_t(pp[2*step]),int32_t(pp[2*step+1])};
                dots=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true,a,true,b,dots,false);
            }
#pragma unroll
            for(int j=0;j<8;++j)accum[tile*8+j]+=float(dots[j])*sp*svs[tile*16+2*j+piece];
        }
        __syncthreads();
    }
    if(live){
#pragma unroll
        for(int j=0;j<128;++j)out[(size_t(row)*16+h)*256+2*j+piece]=bf16(accum[j]/l);
    }
}
template<int KB> int prepare(const void* q,const void* k,const void* v,void* qp,void* qs,void* kp,void* ks,void* vp,void* vs,int tokens,int q_stride0,int q_stride1,void* stream){
    auto s=reinterpret_cast<hipStream_t>(stream);
    prep_qk<true,KB><<<(Q*16+7)/8,256,0,s>>>((const bf16*)q,(uint32_t*)qp,(__half*)qs,tokens,q_stride0,q_stride1);
    prep_qk<false,KB><<<(tokens*2+7)/8,256,0,s>>>((const bf16*)k,(uint32_t*)kp,(__half*)ks,tokens,512,256);
    prep_v<KB><<<dim3(tokens/32,2),256,0,s>>>((const bf16*)v,(uint32_t*)vp,(__half*)vs);
    return int(hipGetLastError());
}
extern "C" __attribute__((visibility("default"))) int iu4_prepare(
    const void* q,const void* k,const void* v,void* qp,void* qs,void* kp,void* ks,void* vp,void* vs,int tokens,int q_stride0,int q_stride1,void* stream){
    if(tokens<Q || tokens>262144 || tokens%32) return int(hipErrorInvalidValue);
    if(tokens<=65536)return prepare<2048>(q,k,v,qp,qs,kp,ks,vp,vs,tokens,q_stride0,q_stride1,stream);
    return prepare<8192>(q,k,v,qp,qs,kp,ks,vp,vs,tokens,q_stride0,q_stride1,stream);
}
extern "C" __attribute__((visibility("default"))) int iu4_attention(
    const void* qp,const void* qs,const void* kp,const void* ks,const void* vp,const void* vs,void* output,int tokens,void* stream){
    if(tokens<Q || tokens>262144 || tokens%32) return int(hipErrorInvalidValue);
    auto s=reinterpret_cast<hipStream_t>(stream);
    if(tokens<=65536)attention<2048><<<dim3((Q+127)/128,16),256,0,s>>>((const uint32_t*)qp,(const __half*)qs,(const uint32_t*)kp,(const __half*)ks,(const uint32_t*)vp,(const __half*)vs,(bf16*)output,tokens);
    else attention<8192><<<dim3((Q+127)/128,16),256,0,s>>>((const uint32_t*)qp,(const __half*)qs,(const uint32_t*)kp,(const __half*)ks,(const uint32_t*)vp,(const __half*)vs,(bf16*)output,tokens);
    return int(hipGetLastError());
}
