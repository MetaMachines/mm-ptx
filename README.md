# MetaMachines
> mm-ptx: PTX Inject, Stack PTX, and AST PTX

## Overview
mm-ptx provides header-only C APIs for working with PTX:

- Stack PTX: a stack-machine instruction language that compiles to valid PTX. It is intended for programmatic generation, mutation, and fast iteration.
- AST PTX: a smaller postorder AST instruction compiler for cases where the upstream system already has an expression tree that yields one f32 value.
- PTX Inject: a CUDA annotation and parsing system that assigns stable PTX register names to variables, then lets you inject custom PTX stubs into compiled PTX.

The main APIs are C99 compliant and live in headers (`stack_ptx.h`, `ast_ptx.h`, and `ptx_inject.h`). CUDA toolchains are only required when compiling and running kernels.

## PTX Inject: CUDA augmentation
Mark a replacement site in CUDA:
```c++
#include <ptx_inject.h>

extern "C"
__global__
void kernel(float* out) {
    float x = 5.0f;
    float y = 3.0f;
    float z = 0.0f;

    PTX_INJECT("func",
        PTX_IN (F32, x),
        PTX_IN (F32, y),
        PTX_OUT(F32, z)
    );

    out[0] = z;
}
```

`PTX_INJECT` is the readable annotation path. Runtime CUDA template generators can emit equivalent inline asm directly, which avoids the macro expansion cost while preserving the same marker contract. Generated PTX stubs from Stack PTX or AST PTX can be inserted at the `func` site.

## Stack PTX: stack program to PTX stub
Stack PTX is useful when generated programs should remain valid after instruction insertion, deletion, or mutation:
```c
#include <stack_ptx.h>
#include <stack_ptx_default_info.h>
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
    stack_ptx_encode_ptx_instruction_mul_ftz_f32,
    stack_ptx_encode_constant_f32(1.0f),
    stack_ptx_encode_ptx_instruction_add_ftz_f32,
    stack_ptx_encode_return
};

static const size_t requests[] = { REG_Z };

// Measure workspace and output size, allocate buffers, then compile.
stack_ptx_compile(...);
```

The register names are the same logical names used by the CUDA injection site. Stack PTX takes bare names such as `"x"` and emits PTX registers such as `%x`:

```ptx
{
.reg .f32 %_s0_<2>;
mul.ftz.f32 %_s0_0, %x, %y;
add.ftz.f32 %_s0_1, %_s0_0, 0f3F800000;
mov.f32 %z, %_s0_1;
}
```

## AST PTX: postorder AST to PTX stub
AST PTX is smaller when an upstream system already has an expression tree that yields one f32 value:

```c
#include <ast_ptx.h>
#include <ast_ptx_instructions.h>

enum { INPUT_X, INPUT_Y, INPUT_COUNT };

const char output_register_name[] = "%z";
const char* const input_register_names[INPUT_COUNT] = {
    [INPUT_X] = "%x",
    [INPUT_Y] = "%y",
};

static const AstPtxInstruction program[] = {
    ast_ptx_encode_input(INPUT_X),
    ast_ptx_encode_input(INPUT_Y),
    ast_ptx_encode_ptx_instruction_mul_ftz_f32,
    ast_ptx_encode_constant(1.0f),
    ast_ptx_encode_ptx_instruction_add_ftz_f32,
    ast_ptx_encode_return
};

// Measure output size, allocate the buffer, then compile.
ast_ptx_compile(...);
```

This emits the same expression with AST PTX temporaries:

```ptx
{
.reg .f32 %ast<2>;
mul.ftz.f32 %ast0, %x, %y;
add.ftz.f32 %ast1, %ast0, 0f3f800000;
mov.f32 %z, %ast1;
}
```

The Stack PTX and AST PTX outputs are PTX stubs. They can be inserted with PTX Inject or embedded directly in generated inline asm.

## Guides
- Stack PTX guide: [STACK_PTX.md](STACK_PTX.md)
- AST PTX guide: [AST_PTX.md](AST_PTX.md)
- PTX Inject guide: [PTX_INJECT.md](PTX_INJECT.md)
- Stack PTX examples: [examples/stack_ptx/README.md](examples/stack_ptx/README.md)
- Stack PTX + PTX Inject examples: [examples/stack_ptx_inject/README.md](examples/stack_ptx_inject/README.md)
- PTX Inject example code: [examples/ptx_inject](examples/ptx_inject)

## Contact
contact@metamachines.co

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this software in your work, please cite it using the following BibTeX entry (generated from the [CITATION.cff](CITATION.cff) file):
```bibtex
@software{Durham_mm-ptx_2025,
  author       = {Durham, Charlie},
  title        = {mm-ptx: PTX Inject and Stack PTX},
  version      = {1.1.1},
  date-released = {2026-03-16},
  url          = {https://github.com/MetaMachines/mm-ptx}
}
```
