# MetaMachines
> mm-ptx: PTX Inject and Stack PTX

## Overview
mm-ptx provides two header-only libraries for working with PTX:

- Stack PTX: a stack-machine instruction language that compiles to valid PTX. It is intended for programmatic generation, mutation, and fast iteration.
- PTX Inject: a CUDA annotation and parsing system that assigns stable PTX register names to variables, then lets you inject custom PTX stubs into compiled PTX.

Both libraries are C99 compliant and live in single headers (`stack_ptx.h` and `ptx_inject.h`). CUDA toolchains are only required when compiling and running kernels.

## PTX Inject: what you write
Mark a site in CUDA with macros:
```c++
#include <ptx_inject.h>

extern "C"
__global__
void kernel(float* out) {
    float x = 5.0f;
    float y = 3.0f;
    float z = 0.0f;
    PTX_INJECT("func",
        PTX_IN (F32, x, x),
        PTX_MOD(F32, y, y),
        PTX_OUT(F32, z, z)
    );
    out[0] = z;
}
```

Compile the CUDA to PTX (nvcc or nvrtc), then use the host-side API to inject a stub:
```c
#include <ptx_inject.h>
#include <stdio.h>
#include <stdlib.h>

const char* annotated_ptx = "..."; // PTX from nvcc/nvrtc

PtxInjectHandle inject;
ptx_inject_create(&inject, annotated_ptx);

size_t inject_idx = 0;
ptx_inject_inject_info_by_name(inject, "func", &inject_idx, NULL, NULL);

const char* reg_x = NULL;
const char* reg_y = NULL;
const char* reg_z = NULL;
ptx_inject_variable_info_by_name(inject, inject_idx, "x", NULL, &reg_x, NULL, NULL, NULL);
ptx_inject_variable_info_by_name(inject, inject_idx, "y", NULL, &reg_y, NULL, NULL, NULL);
ptx_inject_variable_info_by_name(inject, inject_idx, "z", NULL, &reg_z, NULL, NULL, NULL);

char stub[256];
snprintf(
    stub,
    sizeof(stub),
    "\tadd.ftz.f32 %%%2$s, %%%1$s, %%%2$s;\n"
    "\tadd.ftz.f32 %%%3$s, %%%1$s, %%%2$s;",
    reg_x, reg_y, reg_z
);

size_t num_injects = 0;
ptx_inject_num_injects(inject, &num_injects);

const char** stubs = (const char**)calloc(num_injects, sizeof(char*));
stubs[inject_idx] = stub;

size_t out_size = 0;
ptx_inject_render_ptx(inject, stubs, num_injects, NULL, 0, &out_size);

char* out_buffer = (char*)malloc(out_size + 1);
size_t bytes_written = 0;
ptx_inject_render_ptx(inject, stubs, num_injects, out_buffer, out_size + 1, &bytes_written);

free(out_buffer);
free(stubs);
ptx_inject_destroy(inject);
```

This would be equivalent to writing this CUDA kernel directly but without the CUDA to PTX compilation overhead:
```c++
extern "C"
__global__
void kernel(float* out) {
    float x = 5.0f;
    float y = 3.0f;
    float z = 0.0f;
    y = x + y;
    z = x + y;
    out[0] = z;
}
```

## Stack PTX: stack-based instruction compiler
If you do not want to hand-write PTX, you can use Stack PTX to generate the stub:
```c
#include <stack_ptx.h>
#include <stack_ptx_default_info.h>
// Generated from examples/type_descriptions/stack_ptx_descriptions.json
#include "generated/stack_ptx_descriptions.h"

enum { REG_X, REG_Y, REG_Z, REG_COUNT };

static const StackPtxRegister registers[] = {
    [REG_X] = { .name = "x", .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REG_Y] = { .name = "y", .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REG_Z] = { .name = "z", .stack_idx = STACK_PTX_STACK_TYPE_F32 },
};

static const StackPtxInstruction instructions[] = {
    stack_ptx_encode_input(REG_X),
    stack_ptx_encode_input(REG_Y),
    stack_ptx_encode_ptx_instruction_add_ftz_f32,
    stack_ptx_encode_input(REG_X),
    stack_ptx_encode_ptx_instruction_add_ftz_f32,
    stack_ptx_encode_return
};

static const size_t requests[] = { REG_Z };

// Measure output size, allocate buffers, then compile.
stack_ptx_compile(...);
```

This yields a PTX stub you can pass to `ptx_inject_render_ptx`. For a complete runnable example, see [examples/stack_ptx/00_simple/main.c](examples/stack_ptx/00_simple/main.c) and [examples/stack_ptx_inject/00_simple/main.c](examples/stack_ptx_inject/00_simple/main.c).

## Guides
- Stack PTX guide: [STACK_PTX.md](STACK_PTX.md)
- PTX Inject guide: [PTX_INJECT.md](PTX_INJECT.md)
- Stack PTX examples: [examples/stack_ptx/README.md](examples/stack_ptx/README.md)
- Stack PTX + PTX Inject examples: [examples/stack_ptx_inject/README.md](examples/stack_ptx_inject/README.md)
- PTX Inject example code: [examples/ptx_inject](examples/ptx_inject)

## Roadmap
- **OpenMP parallel compile**
  - Demonstrate embarrassingly parallel CPU compilation with OpenMP.
  - Show microsecond-scale Stack PTX stub generation and injection.
  - Do PTX-to-cubin via the nvPTXCompiler static library (No blocking driver compilation).
  - Single core is already up to 25x faster PTX-to-cubin than CUDA-to-cubin on the CuTe GEMM example. With this setup Multicore scales linearly.
- **Networked compile workers**
  - Send Stack PTX stubs over the network to Mac mini machines running Lima containers using nng.
  - Leverage driver-free nvPTXCompiler to compile on low-cost ARM servers with no Nvidia hardware.
  - Fixed-width 8-byte instruction format keeps payloads small after setup, and the cubin is the only returned artifact.
- **Evolutionary search / packed injects**
  - Demonstrate Map-Elites–style evolution of Stack PTX instruction sequences.
  - Pack hundreds or thousands of injection sites per kernel to amortize cuLoadModuleDataEx and launch overhead.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this software in your work, please cite it using the following BibTeX entry (generated from the [CITATION.cff](CITATION.cff) file):
```bibtex
@software{Durham_mm-ptx_2025,
  author       = {Durham, Charlie},
  title        = {mm-ptx: PTX Inject and Stack PTX},
  version      = {1.0.0},
  date-released = {2025-10-19},
  url          = {https://github.com/MetaMachines/mm-ptx}
}
```
