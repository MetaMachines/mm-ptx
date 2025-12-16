#include <ptx_inject.h>

extern "C"
__global__
void
kernel(float* out) {
    float x = 5;
    float y = 3;
    float z;
    PTX_INJECT("func",
        PTX_OUT (F32, v_z, z),
        PTX_MOD (F32, v_x, x),
        PTX_IN  (F32, v_y, y)
    );
    *out = z;
}
    