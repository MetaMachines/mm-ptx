#include <ptx_inject.h>

#define PTX_TYPE_INFO_DOG PTX_TYPES_DESC(f32, f32, f, ID)

extern "C"
__global__
void
kernel(float* out) {
    float x = 5;
    float y = 3;
    float z;
    PTX_INJECT("func",
        PTX_OUT (DOG, z, z),
        PTX_MOD (DOG, x, x),
        PTX_IN  (DOG, y, y)
    );
    *out = z;
}
