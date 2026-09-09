// Copyright (c) Ciru. Grouped G256 prefill with N32 packed banks.
// Routing and WMMA lane ownership adapted from grouped_tilebank_n16_consumer.
#pragma once
namespace routed_direct {
using i32x2=int32_t __attribute__((ext_vector_type(2)));
using i32x8=int32_t __attribute__((ext_vector_type(8)));
struct Descriptor {int32_t expert,first,rows,reserved;};
struct PrefillLayout {size_t counts,offsets,cursors,routes,descriptors,compact,queue,bytes;};
PrefillLayout prefill_layout(size_t cursor,int Tcap){
    PrefillLayout p{};size_t r=size_t(Tcap)*Top,q=(r+15)/16+E;
    auto add=[&](size_t& field,size_t bytes){field=cursor;cursor=align256(cursor+bytes);};
    add(p.counts,E*4);add(p.offsets,(E+1)*4);add(p.cursors,E*4);
    add(p.routes,r*4);add(p.descriptors,q*sizeof(Descriptor));add(p.compact,r*4);add(p.queue,8);
    p.bytes=cursor;return p;
}
__global__ void histogram(const int32_t* ids,int32_t* counts,uint32_t* sticky,int R){
    int route=blockIdx.x*blockDim.x+threadIdx.x;if(route>=R)return;
    int expert=ids[route];
    if(expert>=0&&expert<E)atomicAdd(counts+expert,1);
    else if(expert!= -1)atomicOr(sticky,4u);
}
__global__ void scan_routes(const int32_t* counts,int32_t* offsets,int32_t* cursors){
    int total=0;for(int e=0;e<E;++e){offsets[e]=total;cursors[e]=0;total+=counts[e];}offsets[E]=total;
}
__global__ void scatter(const int32_t* ids,const int32_t* offsets,int32_t* cursors,int32_t* routes,int R){
    int route=blockIdx.x*blockDim.x+threadIdx.x;if(route>=R)return;
    int e=ids[route];if(e<0||e>=E)return;
    routes[offsets[e]+atomicAdd(cursors+e,1)]=route;
}
__global__ void partition(const int32_t* counts,const int32_t* offsets,const int32_t* routes,
        Descriptor* descriptors,int32_t* compact,uint32_t* queue){
    uint32_t nw=0,nc=0;
    for(int e=0;e<E;++e){
        int count=counts[e],first=offsets[e];
        if(count<=4){for(int j=0;j<count;++j)compact[nc++]=routes[first+j];}
        else for(int j=0;j<count;j+=16)descriptors[nw++]={e,first+j,count-j<16?count-j:16,0};
    }
    queue[0]=nw;queue[1]=nc;
}
template<int N,int K,bool Gate,bool A8> __device__ __forceinline__ void wmma_job(
        const uint32_t* codes,const Meta* metadata,const uint32_t* low,const uint32_t* high,
        const __half* scales,const int32_t* sums,float* output,uint32_t* sticky,
        const Descriptor& desc,const int32_t* routes,int tile){
    constexpr int Groups=K/G;
    int lane=threadIdx.x,lane16=lane&15,piece=lane>>4;
    int laneN=16*(tile%2)+lane16;
    bool live=lane16<desc.rows;
    int route=live?routes[desc.first+lane16]:0,row=Gate?route/Top:route;
    size_t bankbase=(size_t(desc.expert)*(N/32)+tile/2)*Groups;
    float values[8]={};
    for(int group=0;group<Groups;++group){
        i32x8 acc_low={},acc_high={};
#pragma unroll
        for(int step=0;step<Words/2;++step){
            int word=group*Words+2*step;
            size_t wi=((bankbase+group)*Words+2*step)*32+laneN;
            i32x2 w={int32_t(codes[wi]),int32_t(codes[wi+32])};
            i32x2 lo={live?int32_t(low[size_t(row)*(K/8)+word]):0,live?int32_t(low[size_t(row)*(K/8)+word+1]):0};
            if constexpr(A8){
                i32x2 hi={live?int32_t(high[size_t(row)*(K/8)+word]):0,live?int32_t(high[size_t(row)*(K/8)+word+1]):0};
                acc_low=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false,w,false,lo,acc_low,false);
                acc_high=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false,w,true,hi,acc_high,false);
            }else acc_low=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false,w,true,lo,acc_low,false);
        }
        uint32_t own_meta=0;
        if(piece==0)own_meta=reinterpret_cast<const uint32_t*>(metadata)[(bankbase+group)*32+laneN];
        uint32_t wm_words[8];
#pragma unroll
        for(int v=0;v<8;++v)wm_words[v]=__shfl(own_meta,2*v+piece,32);
        if(live){
            float sa=__half2float(scales[size_t(row)*Groups+group]);
            float scaled_sum=sa*float(sums[size_t(row)*Groups+group]);
#pragma unroll
            for(int v=0;v<8;++v){
                int dot=A8?acc_low[v]+16*acc_high[v]:acc_low[v];
                Meta wm={__ushort_as_half(uint16_t(wm_words[v])),__ushort_as_half(uint16_t(wm_words[v]>>16))};
                float scaled_dot=sa*float(dot),product=__half2float(wm.scale)*scaled_dot;
                float correction=__half2float(wm.offset)*scaled_sum;
                values[v]=values[v]+(product+correction);
            }
        }
    }
    if(live){
#pragma unroll
        for(int v=0;v<8;++v){
            float value=values[v];output[size_t(route)*N+tile*16+2*v+piece]=value;
            if(!isfinite(value))atomicOr(sticky,8u);
        }
    }
}
template<int N,int K,bool Gate,bool A8> __global__ __launch_bounds__(32) void grouped_projection(
        const int32_t* ids,const uint32_t* codes,const Meta* metadata,
        const uint32_t* low,const uint32_t* high,const __half* scales,const int32_t* sums,
        float* output,uint32_t* sticky,const Descriptor* descriptors,const int32_t* routes,
        const int32_t* compact,const uint32_t* queue){
    constexpr int Tiles=N/16,SparseTiles=N/32;
    uint32_t nw=queue[0],nc=queue[1],wjobs=nw*Tiles,jobs=wjobs+nc*SparseTiles;
    for(uint32_t job=blockIdx.x;job<jobs;job+=gridDim.x){
        if(job<wjobs)wmma_job<N,K,Gate,A8>(codes,metadata,low,high,scales,sums,output,sticky,descriptors[job/Tiles],routes,job%Tiles);
        else{uint32_t local=job-wjobs;projection_n32_job<N,K,Gate,A8>(ids,codes,metadata,low,high,scales,sums,output,sticky,compact[local/SparseTiles],local%SparseTiles);}
    }
}
} // namespace routed_direct
