extern "C"
__global__
void
kernel(float* out) {
    float x = 5;
    float y = 3;
    float z;
    /* PTX_INJECT func
        in dog x
        mod dog y
        out dog z
    */
    *out = z;
}
