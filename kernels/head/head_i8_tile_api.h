// PHYSICAL ABI: weights are UINT32[N/16,K/16,4,16], low-K signed byte first.
// Scales remain FP16[N,16]. No row-major bank may be passed.
// Copyright2026 Ciru. Separate fixed-head signed I8 experiment, ABI1.
#ifndef CIRU_ORNITH_HEAD_I8_API_H
#define CIRU_ORNITH_HEAD_I8_API_H
#include <stddef.h>
#include <stdint.h>
#include <hip/hip_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif
#define ORNITH_HEAD_I8_EXPORT __attribute__((visibility("default")))
typedef struct OrnithHeadI8Layout {
    size_t workspace_bytes, transformed, activation_words, activation_scales, projection;
} OrnithHeadI8Layout;
enum OrnithHeadI8Error {
    ORNITH_HEAD_I8_NONFINITE_INPUT_TRANSFORM=1,
    ORNITH_HEAD_I8_INVALID_ACTIVATION=2,
    ORNITH_HEAD_I8_NONFINITE_PROJECTION=4,
    ORNITH_HEAD_I8_NONFINITE_BF16=8
};
// Fixed K2048/G128/H128; N=256 synthetic or248320 real vocabulary only.
// 0<=M<=Mcap<=64. This prototype capacity is not a product concurrency limit.
// Input BF16[M,2048], row-major signed INT8 weight[N,2048], code range[-127,127],
// FP16 positive scale[N,16] with floor2^-14. Caller seals immutable bank values.
// No affine offset. Activation uses FP32 H128 (7ascending butterflies,
// reciprocal factor bits0x3db504f3), RN(absmax/127), max2^-14 then FP16RNE,
// zero-group scale1/codes0; codes RN(divideRN(value,stored_scale)) clamp+-127.
// Each exact INT32 G128 dot is bounded2,064,512; group0..15 accumulated FP32:
// scaled=sa*float(dot); term=sw*scaled; result=result+term; no FMA contraction.
// Output BF16RNE[M,N]. Diagnostic FP32[Mcap,N] lives in caller workspace.
// UINT32[Mcap,512] activation_words pack4 signed bytes, lowest K byte first.
// Optional group_dots INT32[M,N,16] stores raw group dots without changing math.
// group_dots may be null only when group_dots_bytes==0; then those stores vanish
// through a separate compile-time specialization. It is diagnostic, not needed
// by the connected runtime or component timing path.
// geometry0=WMMA M16/N16;1=DOT4 M1/N8;2=compact iffM<=4 (correctness boundary,
// not a measured crossover). External UINT32[1] flags are sticky, never reset.
// M0 validates scalars and optional-buffer size contract then touches no memory.
// No allocation/sync/host tensor reads/default stream. Mutable buffers cannot
// alias any other range. Alignment: workspace256,weights/words/dots/flags4,
// input/scales/output2. Native errors make outputs unusable.
ORNITH_HEAD_I8_EXPORT uint32_t ornith_head_i8_tile_abi_version(void);
ORNITH_HEAD_I8_EXPORT hipError_t ornith_head_i8_tile_get_layout(int Mcap,int N,OrnithHeadI8Layout* out);
ORNITH_HEAD_I8_EXPORT hipError_t ornith_head_i8_tile_launch(
    const void* input_bf16,const void* weights_i8,const void* scales_f16,
    void* workspace,size_t workspace_bytes,void* output_bf16,
    void* group_dots_i32,size_t group_dots_bytes,void* flags_u32,
    int M,int Mcap,int N,int geometry,hipStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
