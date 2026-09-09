// Ciru: deterministic FP32 exponential for connected quantization semantics.
// Mathematical Taylor coefficients, no fitted tables or vendor exp primitive.
// Build with -ffp-contract=off; explicit F32 operations match portable_exp.py.
#pragma once

__device__ __forceinline__ float ornith_portable_exp(float x) {
    if (isnan(x)) return x;
    if (x <= -104.f) return 0.f;
    if (x >= 89.f) return INFINITY;
    const float n_float = nearbyintf(x * 0x1.715476p+0f); // 0x3fb8aa3b
    const int n = static_cast<int>(n_float);
    const float high = n_float * 0x1.62e400p-1f; // 0x3f317200
    const float low = n_float * 0x1.7f7d1cp-20f; // 0x35bfbe8e
    const float reduced_high = x - high;
    const float r = reduced_high - low;
    float p = 0x1.a01a02p-16f; // 1/8!
    p = p * r + 0x1.a01a02p-13f; // 1/7!
    p = p * r + 0x1.6c16c2p-10f; // 1/6!
    p = p * r + 0x1.111112p-7f;  // 1/5!
    p = p * r + 0x1.555556p-5f;  // 1/4!
    p = p * r + 0x1.555556p-3f;  // 1/3!
    p = p * r + 0x1p-1f;
    p = p * r + 0x1p+0f;
    p = p * r + 0x1p+0f;
    return ldexpf(p, n);
}
