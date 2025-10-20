extern "C"
__global__
void
kernel(float* out) {
    float x = 5;
    float y = 3;
    float z;
    /* PTX_INJECT func
        in f32 x
        mod f32 y
        out f32 z
    */
    *out = z;
}
