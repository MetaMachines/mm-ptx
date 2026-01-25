#include <ptx_inject.h>
#include "philox.cuh"

#include <stdio.h>
#include <stdint.h>

#define PTX_TYPE_INFO_PHILOX PTX_TYPES_DESC(u32, u32, r, ID)

#define PTX_PHILOX(state)                           \
    PTX_IN (PHILOX, philox_key_x, state.key.x),     \
    PTX_IN (PHILOX, philox_key_y, state.key.y),     \
    PTX_MOD (PHILOX, philox_ctr_x, state.ctr.x),    \
    PTX_IN (PHILOX, philox_ctr_y, state.ctr.y),     \
    PTX_IN (PHILOX, philox_ctr_z, state.ctr.z),     \
    PTX_IN (PHILOX, philox_ctr_w, state.ctr.w)

extern "C"
__global__
void
kernel(
    unsigned int* out
) {
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;
    unsigned int w = 0;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t state = philox_init(0,1234,tid);

    uint4 output;

    output = philox_curand4(&state);
    output = philox_curand4(&state);
    state.ctr.x = 0;

    PTX_INJECT("func",
        PTX_PHILOX(state),
        PTX_OUT (U32, x, x),
        PTX_OUT (U32, y, y),
        PTX_OUT (U32, z, z),
        PTX_OUT (U32, w, w)
    );

    printf("%u\n", state.ctr.x);
    printf("%u\n", x);
    printf("%u\n", y);
    printf("%u\n", z);
    printf("%u\n", w);

    printf("%u\n", output.x);
    printf("%u\n", output.y);
    printf("%u\n", output.z);
    printf("%u\n", output.w);

    output = philox_curand4(&state);
    output = philox_curand4(&state);
    state.ctr.x = 2;

    PTX_INJECT("func",
        PTX_PHILOX(state),
        PTX_OUT (U32, x, x),
        PTX_OUT (U32, y, y),
        PTX_OUT (U32, z, z),
        PTX_OUT (U32, w, w)
    );

    printf("%u\n", state.ctr.x);
    printf("%u\n", x);
    printf("%u\n", y);
    printf("%u\n", z);
    printf("%u\n", w);

    printf("%u\n", output.x);
    printf("%u\n", output.y);
    printf("%u\n", output.z);
    printf("%u\n", output.w);

    *out = z;
}
