
extern "C"
__global__
void
kernel_000000(float* out) {
    float x = 5;
    float y = 3;
    float z;
    /* PTX_INJECT func_000000
        in f32 x
        mod f32 y
        out f32 z
    */
    *out = z;
}
