#include <cuda_fp16.h>

extern "C"
__global__
void
kernel(float* out) {
    __half2 x_y =  __floats2half2_rn(3, 5);
    __half z = __float2half_rn(4);
    float w;
    /* PTX_INJECT func
        in f16X2 x_y
        in f16 z
        out f32 w
    */
    *out = w;
}
