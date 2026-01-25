# mm-ptx compile bench

This benchmark compares compile time to cubin for two generator paths:

1. PTX inject: CUDA -> PTX (NVRTC), Stack-PTX stubs injected, PTX -> cubin (nvPTXCompiler)
2. CUDA inline PTX: Stack-PTX stubs embedded as inline PTX, CUDA -> cubin (NVRTC)

The kernel layout matches the app_elites_nle tiling (num kernels, groups per kernel, tile size).
The benchmark does not load cubins. Stack-PTX stubs and per-module sources are precomputed outside the
timed region; timings cover only PTX->cubin and CUDA->cubin compilation. OpenMP is optional via `--cores`.

## Build

From the repo root:

```
cmake -S . -B build -DMMPTX_BUILD_BENCH=ON
cmake --build build --target mm_ptx_compile_bench
```

## Run

```
./build/mm-ptx/bench/mm_ptx_compile_bench --modules 32 --kernels 2 --groups-per-kernel 16 --ptx-instructions-per-program 32
```

Override SM targets with `--sm` (both) or `--sm-ptx`/`--sm-cubin` (e.g., `--sm 80`).

Additional options:

- `--sm-ptx` and `--sm-cubin` let you target NVRTC PTX and nvptxcompiler SMs independently (default `8.0`).
- `--cores` controls OpenMP threads (0 = all cores, default runtime value).
- `--workspace-bytes` controls per-thread scratch arena size (default: auto).
- `--ptx-instructions-per-program` controls stack-ptx instruction count per program (includes return).
- `--dump-ptx-cu` dumps the CUDA source used for the PTX-inject path (alias: `--dump-cu`).
- `--dump-ptx` dumps the base PTX emitted by NVRTC before injection.
- `--dump-module-ptx` / `--dump-module-cu` dump one module's generated sources for inspection.
- `--dump-module-index` selects which module to dump (default: 0).
