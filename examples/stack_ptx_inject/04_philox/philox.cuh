#pragma once

#include <stdint.h>

#define QUALIFIERS static __forceinline__ __device__

#define PHILOX_W32_0   (0x9E3779B9)
#define PHILOX_W32_1   (0xBB67AE85)
#define PHILOX_M4x32_0 (0xD2511F53)
#define PHILOX_M4x32_1 (0xCD9E8D57)

#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)

QUALIFIERS 
uint4 
_philox4x32round(
    uint4 ctr, 
    uint2 key
) {
   unsigned int hi0 = __umulhi(PHILOX_M4x32_0, ctr.x);
   unsigned int hi1 = __umulhi(PHILOX_M4x32_1, ctr.z);
   unsigned int lo0 = PHILOX_M4x32_0 * ctr.x;
   unsigned int lo1 = PHILOX_M4x32_1 * ctr.z;

   return {hi1^ctr.y^key.x, lo1, hi0^ctr.w^key.y, lo0};
}

QUALIFIERS 
uint4 
_curand_Philox4x32_10(
    uint4 c, 
    uint2 k
) {
   c = _philox4x32round(c, k);                           // 1 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 2
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 3 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 4 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 5 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 6 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 7 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 8 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 9 
   k.x += PHILOX_W32_0; k.y += PHILOX_W32_1;
   c = _philox4x32round(c, k);                           // 10
   return c;
}

QUALIFIERS 
float4 
_curand_uniform4(
    uint4 x
) {
    float4 y;
    y.x = x.x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.y = x.y * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.z = x.z * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.w = x.w * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    return y;
}

QUALIFIERS 
float2 
_curand_box_muller(
    unsigned int x, 
    unsigned int y
) {
    float2 result;
    float u = x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2);
    float v = y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI/2);
    float s;

    s = sqrtf(-2.0f * logf(u));
    __sincosf(v, &result.x, &result.y);

    result.x *= s;
    result.y *= s;
    return result;
}

struct curandStatePhilox4_32_10 {
   uint4 ctr;
   uint2 key;
};

typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

__device__
curandStatePhilox4_32_10_t
philox_init(
    uint64_t launch_id,
    uint64_t seed,
    uint32_t utid
) {
    uint4 ctr = {0, (unsigned int)launch_id, (unsigned int)(launch_id >> 32), utid};
    uint2 key = {(unsigned int)seed, (unsigned int)(seed>>32)};
    return {ctr, key};
}

QUALIFIERS 
uint4 
philox_curand4(
    curandStatePhilox4_32_10_t *state
) {    
    uint4 output = _curand_Philox4x32_10(state->ctr,state->key);
    state->ctr.x++;
    return output;
}

QUALIFIERS 
float4 
philox_curand_uniform4(
    curandStatePhilox4_32_10_t *state
) {
   return _curand_uniform4(philox_curand4(state));
}

QUALIFIERS 
float4 
philox_curand_normal4(
    curandStatePhilox4_32_10_t *state
) {
    float4 result;
    float2 _result;
    uint4 x = philox_curand4(state);
    _result = _curand_box_muller(x.x, x.y);
    result.x = _result.x;
    result.y = _result.y;
    _result = _curand_box_muller(x.z, x.w);
    result.z = _result.x;
    result.w = _result.y;
    return result;
}
