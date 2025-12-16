#include <ptx_inject.h>
#include <cuda_fp16.h>

extern "C"
__global__
void
kernel(float* out) {
    __half2 x_y =  __floats2half2_rn(3, 5);
    __half z = __float2half_rn(4);
    float w;
    PTX_INJECT("func",
        PTX_IN(F16X2, x_y, x_y),
        PTX_IN(F16, z, z),
        PTX_OUT(F32, w, w)
    );
    *out = w;
}
