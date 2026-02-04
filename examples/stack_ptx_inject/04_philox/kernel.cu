#include <ptx_inject.h>
#include <philox.cuh>

#include <stdio.h>
#include <stdint.h>

extern "C"
__global__
void
kernel(
    float* out
) {
    float x = 0;
    float y = 0;
    float z = 0;
    float w = 0;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t state = philox_init(0,1234,tid);

    float4 output;

    output = philox_curand_normal4(&state);
    output = philox_curand_normal4(&state);
    state.ctr.x = 0;

    PTX_INJECT("func",
        PTX_PHILOX(state),
        PTX_OUT (F32, x, x),
        PTX_OUT (F32, y, y),
        PTX_OUT (F32, z, z),
        PTX_OUT (F32, w, w)
    );

    printf("%u\n", state.ctr.x);
    printf("%f:%f\n", x, output.x);
    printf("%f:%f\n", y, output.y);
    printf("%f:%f\n", z, output.z);
    printf("%f:%f\n", w, output.w);

    output = philox_curand_normal4(&state);
    output = philox_curand_normal4(&state);
    state.ctr.x = 2;

    PTX_INJECT("func",
        PTX_PHILOX(state),
        PTX_OUT (F32, x, x),
        PTX_OUT (F32, y, y),
        PTX_OUT (F32, z, z),
        PTX_OUT (F32, w, w)
    );

    printf("%u\n", state.ctr.x);
    printf("%f:%f\n", x, output.x);
    printf("%f:%f\n", y, output.y);
    printf("%f:%f\n", z, output.z);
    printf("%f:%f\n", w, output.w);

    *out = z;
}
