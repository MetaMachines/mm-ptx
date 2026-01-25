# mm-ptx compile bench

This benchmark compares compile time to cubin for three generator paths:

1. CUDA-only individuals to cubin
2. Stack PTX inject individuals to cubin
3. CUDA inline-PTX individuals (from Stack PTX stubs) to cubin

The kernel layout matches the app_elites_nle tiling (num kernels, groups per kernel, tile size).
The benchmark is single-threaded and does not load cubins.

## Build

From the repo root:

```
cmake -S . -B build -DMMPTX_BUILD_BENCH=ON
cmake --build build --target mm_ptx_compile_bench
```

## Run

```
./build/mm-ptx/bench/mm_ptx_compile_bench --modules 32 --num-kernels 2 --groups-per-kernel 16 --gene-length 32
```

If automatic SM detection fails, pass `--sm` (e.g., `--sm 80`).
