# Inject Site Design Notes

This document captures patterns for compile-time PTX inject site naming, reusable inject signatures, and NVRTC template instantiation.

## Goals

- Define an inject signature once (for example `F32 -> F32`).
- Reuse it across many PTX sites.
- Keep stable, searchable site names in generated PTX.
- Support many kernels with many sites each (for example `4 * 32 = 128`).
- Keep host-side mapping simple and deterministic.

## 1) Generic Template Inject With Compile-Time Site ID

This keeps the inject logic generic and emits `SiteId` into PTX comments so parsing can recover site identity.

```cpp
template<int SiteId>
static __device__ __forceinline__ float inject_f32_to_f32(float x) {
  static_assert(SiteId >= 0, "SiteId must be compile-time");

  float y;
  asm volatile(
    "{\n\t"
    ".reg .f32 %%_x0;\n\t"                  // out
    ".reg .f32 %%_x1;\n\t"                  // in
    "mov.f32 %%_x1, %1;\n\t"
    "mov.f32 %%_x0, %%_x1;\n\t"             // passthrough default
    "// PTX_INJECT_START f32_to_f32_%2\n\t" // SiteId appears in PTX
    "// _x0 o f32 F32 y\n\t"
    "// _x1 i f32 F32 x\n\t"
    "// PTX_INJECT_END\n\t"
    "mov.f32 %0, %%_x0;\n\t"
    "}\n\t"
    : "=f"(y)
    : "f"(x), "n"(SiteId));                 // compile-time immediate
  return y;
}
```

Notes:

- `"n"` requires compile-time constants.
- Site IDs are visible in PTX marker comments.

## 2) Many Kernels, Many Sites, One Signature

Example shape:

- `NumKernels = 4`
- `SitesPerKernel = 32`
- Global site id: `global = kernel_id * SitesPerKernel + local_site_id`

This version also models common output layout: same input value used for all sites, each site writes to its own output lane.

```cpp
#include <utility>

template<int NumKernels, int SitesPerKernel, int VariantId = 0>
struct InjectProgram {
  template<int KernelId, int LocalSiteId>
  __device__ static __forceinline__ float inject(float x) {
    static_assert(KernelId >= 0 && KernelId < NumKernels, "bad KernelId");
    static_assert(LocalSiteId >= 0 && LocalSiteId < SitesPerKernel, "bad LocalSiteId");
    constexpr int GlobalSiteId = KernelId * SitesPerKernel + LocalSiteId;

    float y;
    asm volatile(
      "{\n\t"
      ".reg .f32 %%_x0;\n\t"
      ".reg .f32 %%_x1;\n\t"
      "mov.f32 %%_x1, %1;\n\t"
      "mov.f32 %%_x0, %%_x1;\n\t"
      "// PTX_INJECT_START f32_to_f32_k%2_s%3_g%4_v%5\n\t"
      "// _x0 o f32 F32 y\n\t"
      "// _x1 i f32 F32 x\n\t"
      "// PTX_INJECT_END\n\t"
      "mov.f32 %0, %%_x0;\n\t"
      "}\n\t"
      : "=f"(y)
      : "f"(x), "n"(KernelId), "n"(LocalSiteId), "n"(GlobalSiteId), "n"(VariantId));
    return y;
  }

  template<int KernelId, int... I>
  __device__ static __forceinline__ void emit_sites(
      float x,
      int sample_idx,
      float* out_site_major,
      int out_stride,
      std::integer_sequence<int, I...>) {
    int unused[] = {
      0,
      (out_site_major[(KernelId * SitesPerKernel + I) * out_stride + sample_idx] =
           inject<KernelId, I>(x),
       0)...};
    (void)unused;
  }

  template<int KernelId>
  __global__ static void kernel(const float* in, int n, float* out_site_major, int out_stride) {
    int sample_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (sample_idx >= n) return;
    float x = in[sample_idx];
    emit_sites<KernelId>(
        x, sample_idx, out_site_major, out_stride,
        std::make_integer_sequence<int, SitesPerKernel>{});
  }
};
```

Output layout for `out_site_major`:

- Index = `global_site * out_stride + sample_idx`
- Shape = `[total_sites][num_samples]`

## 3) Keep Template Logic Fixed, Generate Only Small Wrapper Layer

Keep the big template code handwritten once. Generator emits only config and exported kernel symbols.

```cpp
using P = InjectProgram<4, 32, 0>;

extern "C" __global__ void kernel_0(const float* in, int n, float* out, int stride) {
  P::template kernel<0>(in, n, out, stride);
}
extern "C" __global__ void kernel_1(const float* in, int n, float* out, int stride) {
  P::template kernel<1>(in, n, out, stride);
}
extern "C" __global__ void kernel_2(const float* in, int n, float* out, int stride) {
  P::template kernel<2>(in, n, out, stride);
}
extern "C" __global__ void kernel_3(const float* in, int n, float* out, int stride) {
  P::template kernel<3>(in, n, out, stride);
}
```

This is usually the cleanest split between reusable logic and generated boilerplate.

## 4) NVRTC Lowered Names For Template Instantiations (No Kernel Wrapper Generator Required)

You can request templated kernels directly with name expressions.

```cpp
std::vector<std::string> exprs;
for (int k = 0; k < 4; ++k) {
  exprs.push_back("InjectProgram<4,32,0>::kernel<" + std::to_string(k) + ">");
  NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, exprs.back().c_str()));
}

NVRTC_SAFE_CALL(nvrtcCompileProgram(prog, num_opts, opts));

for (const std::string& e : exprs) {
  const char* lowered = nullptr;
  NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, e.c_str(), &lowered));

  CUfunction fn = nullptr;
  CU_SAFE_CALL(cuModuleGetFunction(&fn, module, lowered));
  // cache fn and launch later
}
```

Rules:

- Add name expressions before `nvrtcCompileProgram`.
- Use exact same expression string for `nvrtcGetLoweredName`.
- Template args must be compile-time constants.

## 5) Overhead Expectations

- Runtime overhead: none from using template parameters or lowered names.
- Compile-time overhead: scales with number of instantiated sites/kernels/variants.
- Binary/PTX size: also scales with instantiations.

For large site counts (for example 1024), template-based NVRTC and generated-source approaches are typically similar if they produce similar final code volume.

## 6) Practical Recommendation

- Keep one fixed template core (section 2).
- Generate only:
  - config tuple (`NumKernels`, `SitesPerKernel`, `VariantId`)
  - exported wrappers or NVRTC name-expression list.
- Standardize marker naming (`k%u_s%u_g%u_v%u`) so host parsing is deterministic.
- Keep register metadata stable (`_x0`, `_x1`, ...) per signature.
