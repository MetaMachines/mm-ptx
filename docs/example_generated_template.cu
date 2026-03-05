#include <utility>  // std::integer_sequence, std::make_integer_sequence

namespace {  // anonymous namespace => internal linkage (no .visible)

template<int SitesPerKernel, int KernelId, int LocalSiteId>
__device__ __forceinline__ float inject_site(float x) {
  static_assert(LocalSiteId >= 0 && LocalSiteId < SitesPerKernel, "LocalSiteId out of range");

  constexpr int GlobalSiteId = KernelId * SitesPerKernel + LocalSiteId;
  float y;

  asm volatile(
    "{\n\t"
    ".reg .f32 %%_x0;\n\t"
    ".reg .f32 %%_x1;\n\t"
    "mov.f32 %%_x1, %1;\n\t"
    "// PTX_INJECT f32_to_f32_k%2_s%3_g%4\n\t"
    // "// _x0 o f32 F32 y\n\t"
    // "// _x1 i f32 F32 x\n\t"
    // "// PTX_INJECT_END\n\t"
    "mov.f32 %0, %%_x0;\n\t"
    "}\n\t"
    : "=f"(y)
    : "f"(x),
      "n"(KernelId),
      "n"(LocalSiteId),
      "n"(GlobalSiteId)
  );

  return y;
}

template<int SitesPerKernel,
         int KernelId, int... I>
__device__ __forceinline__ void emit_sites(
    float x, 
    int sample, 
    float* out, 
    int stride, 
    std::integer_sequence<int, I...>
) {

  int unused[] = {
    0,
    ( 
        (out[(KernelId * SitesPerKernel + I) * stride + sample] = inject_site<SitesPerKernel, KernelId, I>(x)),
    0
    )...
  };
  (void)unused;
}

template<int SitesPerKernel, int KernelId>
__device__ __forceinline__ void run_kernel(const float* in, int n, float* out, int stride) {

  int sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= n) return;

  float x = in[sample];
  emit_sites<SitesPerKernel, KernelId>(x, sample, out, stride, std::make_integer_sequence<int, SitesPerKernel>{});
}

} // anonymous namespace

// generated (small)
extern "C" __global__ void kernel_0(const float* in, int n, float* out, int stride) { run_kernel<32, 0>(in, n, out, stride); }
extern "C" __global__ void kernel_1(const float* in, int n, float* out, int stride) { run_kernel<32, 1>(in, n, out, stride); }
extern "C" __global__ void kernel_2(const float* in, int n, float* out, int stride) { run_kernel<32, 2>(in, n, out, stride); }
extern "C" __global__ void kernel_3(const float* in, int n, float* out, int stride) { run_kernel<32, 3>(in, n, out, stride); }