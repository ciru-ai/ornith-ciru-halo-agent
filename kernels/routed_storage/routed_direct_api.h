// Copyright (c) Ciru. Exploratory fixed H128/W4-G256/A8 direct routed ABI.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <hip/hip_runtime_api.h>
#define ORNITH_ROUTED_DIRECT_EXPORT __attribute__((visibility("default")))
// Codes [E,N/16,K/256,32,16]; metadata half [E,N/16,K/256,16,2].
// Actual expert IDs. Tcap<=2048, Top8, H2048, I512. T<=8 uses direct DOT8;
// T>8 uses GPU grouping with WMMA for expert counts>4 and DOT8 otherwise.
// Layout fields and offsets preserve the direct ABI; workspace_bytes also
// includes internal routing scratch when Tcap>8. Healthy finite panel only;
// negative padding IDs skip their expert payload. No production admission claim.
struct OrnithRoutedDirectLayout {
    size_t workspace_bytes,gate_x,gate_low,gate_high,gate_scales,gate_sums,gate_y;
    size_t middle,down_x,down_low,down_high,down_scales,down_sums,route_out,final_f32;
};
extern "C" {
ORNITH_ROUTED_DIRECT_EXPORT hipError_t ornith_routed_direct_get_layout(int Tcap,OrnithRoutedDirectLayout* out);
ORNITH_ROUTED_DIRECT_EXPORT hipError_t ornith_routed_direct_launch(
    const void* input_bf16,const void* route_weights_f32,const void* ids_i32,
    const void* gate_codes,const void* gate_meta,const void* down_codes,const void* down_meta,
    void* workspace,size_t bytes,void* output_bf16,void* flags_u32,int T,int Tcap,hipStream_t stream);
// Experimental signed A4 activations at both expert projections. Identical
// G256 weight/layout ABI; high-nibble scratch is reserved but unused.
ORNITH_ROUTED_DIRECT_EXPORT hipError_t ornith_routed_direct_launch_a4(
    const void* input_bf16,const void* route_weights_f32,const void* ids_i32,
    const void* gate_codes,const void* gate_meta,const void* down_codes,const void* down_meta,
    void* workspace,size_t bytes,void* output_bf16,void* flags_u32,int T,int Tcap,hipStream_t stream);
// Experimental prefill policy: A4 only for T>64, otherwise unchanged A8.
ORNITH_ROUTED_DIRECT_EXPORT hipError_t ornith_routed_direct_launch_a4_prefill(
    const void* input_bf16,const void* route_weights_f32,const void* ids_i32,
    const void* gate_codes,const void* gate_meta,const void* down_codes,const void* down_meta,
    void* workspace,size_t bytes,void* output_bf16,void* flags_u32,int T,int Tcap,hipStream_t stream);
}
