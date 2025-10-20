extern "C"
__global__
void
kernel(float* out) {
    float d0, d1, d2, d3;
    /* PTX_INJECT func
        out f32 d0
        out f32 d1
        out f32 d2
        out f32 d3
    */
    out[0 * 32 + threadIdx.x] = d0;
    out[1 * 32 + threadIdx.x] = d1;
    out[2 * 32 + threadIdx.x] = d2;
    out[3 * 32 + threadIdx.x] = d3;
}
