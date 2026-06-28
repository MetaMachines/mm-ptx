# AST PTX

## Overview
AST PTX is a small C99 header API for compiling a postorder AST instruction array into a single f32 PTX expression. It is meant for cases where the upstream system already has a sane AST and only needs a direct PTX stub, without Stack PTX's mutation-friendly stack-program layer.

The core compiler lives in `ast_ptx.h`. Concrete f32 instruction encodes and the CPU interpreter live in:

- `tools/ast_ptx_instructions.h`
- `tools/ast_ptx_interpreter.h`

Define `AST_PTX_IMPLEMENTATION` in exactly one translation unit:

```c
#define AST_PTX_IMPLEMENTATION
#include <ast_ptx_interpreter.h>
```

## Shape
AST programs are arrays of `AstPtxInstruction` terminated by `ast_ptx_encode_return`. Leaves push inputs, constants, or routine arguments. PTX instruction encodes consume the previous values and produce one value. A top-level return moves that final value into the requested output register.

```c
static const AstPtxInstruction program[] = {
    ast_ptx_encode_input(0u),
    ast_ptx_encode_input(1u),
    ast_ptx_encode_ptx_instruction_mul_ftz_f32,
    ast_ptx_encode_constant(1.0f),
    ast_ptx_encode_ptx_instruction_add_ftz_f32,
    ast_ptx_encode_return
};
```

Use the usual measure-and-allocate flow:

```c
size_t required = 0;
ast_ptx_compile("%z", inputs, 2,
                ast_ptx_ptx_instruction_names,
                ast_ptx_ptx_instruction_num_args,
                AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
                routines, num_routines,
                program,
                NULL, 0, &required);

char* stub = malloc(required + 1);
ast_ptx_compile("%z", inputs, 2,
                ast_ptx_ptx_instruction_names,
                ast_ptx_ptx_instruction_num_args,
                AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
                routines, num_routines,
                program,
                stub, required + 1, &required);
```

The generated stub can be passed to PTX Inject. The interpreter is useful for checking that a generated AST produces the same scalar result on the CPU.

## Routines
Routines are also postorder AST instruction arrays. They read their call arguments with `ast_ptx_encode_routine_arg(idx)` and return one value. This is intended for reusable operations such as `pow`, `log10`, or guarded math routines while keeping the top-level AST compiler simple.
