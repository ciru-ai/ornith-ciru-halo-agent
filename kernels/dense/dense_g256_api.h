// Copyright (c) Ciru. Standalone Ornith dense IU4 experiment, ABI1.
#ifndef CIRU_ORNITH_DENSE_G256_API_H
#define CIRU_ORNITH_DENSE_G256_API_H
#include <stddef.h>
#include <stdint.h>
#include <hip/hip_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif
#define ORNITH_DENSE_G256_EXPORT __attribute__((visibility("default")))
typedef struct OrnithDenseG256Layout {
    size_t workspace_bytes, transformed, low, high, scales, sums, projection;
} OrnithDenseG256Layout;
enum OrnithDenseG256Error {
    ORNITH_DENSE_G256_NONFINITE_INPUT_TRANSFORM=1,
    ORNITH_DENSE_G256_INVALID_ACTIVATION=2,
    ORNITH_DENSE_G256_NONFINITE_PROJECTION=4,
    ORNITH_DENSE_G256_NONFINITE_BF16=8
};
// Runtime N: positive multiple16 <=12288; K: positive multiple256 <=8192.
// 0<=M<=Mcap<=8192. Input/output are contiguous GPU BF16 tensors.
// Transform is exactly H128; bits is8. Geometry0 WMMA M16/N16;
// geometry1 compact M1/N16; geometry2 selects compact iff M<=4.
// G=256. This retains the measured QKVZ arithmetic for dense model families.
// Input BF16[M,K], codes U32[N/16,K/G,G/8,16] (eight low-first K nibbles/word),
// metadata FP16[N/16,K/G,16,2] interleaved scale/offset. Caller seals finite metadata
// with scale>=2^-14 before use; launch does not scan the immutable bank.
// Output BF16[M,N]. Workspace contains FP32[Mcap,K] transformed input,
// UINT32[Mcap,K/8] low/high planes, FP16 activation scales
// and INT32 sums [Mcap,K/G], and diagnostic FP32[Mcap,N] projection.
// All workspace offsets256 aligned. No expert dimension/routing/alias banks.
// H128 butterfly FP32, factor0x3db504f3. G-element activation absmax/qmax uses
// correctly rounded FP32 division; max(scale,2^-14) then FP16 RNE; zero groups
// scale1. Signed q round-nearest-even/clamp +-127. A8 low unsigned and
// high signed uses two IU4 products. For each ascending K group, FP32 tree:
// scaled_dot=sa*dot; scaled_sum=sa*sumq; product=sw*scaled_dot;
// correction=offset*scaled_sum; accumulator=accumulator+(product+correction).
// No FMA contraction. Final BF16 RNE, with FP32 diagnostic retained.
// Error flags UINT32[1] are sticky and never cleared, including M0. Invalid
// numerical output is unusable. M0 returns after scalar argument validation
// without touching any pointer. Caller owns lifetime and completion checks.
// No allocation, synchronization, host data read or default-stream use.
// Workspace/output/flags must not alias any other range; read/read alias is
// allowed. Minimum alignment: workspace256, codes/meta/flags4, input/output2.
ORNITH_DENSE_G256_EXPORT uint32_t ornith_dense_g256_abi_version(void);
ORNITH_DENSE_G256_EXPORT hipError_t ornith_dense_g256_get_layout(int Mcap,int N,int K,int bits,OrnithDenseG256Layout* out);
ORNITH_DENSE_G256_EXPORT hipError_t ornith_dense_g256_launch(
    const void* input_bf16,const void* codes,const void* metadata,
    void* workspace,size_t workspace_bytes,void* output_bf16,void* flags_u32,
    int M,int Mcap,int N,int K,int bits,int transform,int geometry,hipStream_t stream);
// Prefill preparation: H128 in FP32 rounded to BF16[M,K], and W4 metadata
// dequantized to BF16[N,K] in the same transformed basis. Caller then computes
// transformed_bf16 @ weights_bf16.T. This uses BF16 activations instead of A8.
// Same shape/alignment/nonalias/sticky rules, except output alignment is2 and
// no workspace is used. No allocation, synchronization, or host data read.
ORNITH_DENSE_G256_EXPORT hipError_t ornith_dense_g256_prepare_bf16(
    const void* input_bf16,const void* codes,const void* metadata,
    void* transformed_bf16,void* weights_bf16,void* flags_u32,
    int M,int N,int K,hipStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
