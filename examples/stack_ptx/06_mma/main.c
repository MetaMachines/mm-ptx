/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

// Import Stack PTX for this compilation using STACK_PTX_IMPLEMENTATION
#define STACK_PTX_DEBUG
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_default_generated_types.h>
#include <stack_ptx_default_info.h>

#include <check_result_helper.h>

typedef enum {
    REGISTER_INPUT_0,
    REGISTER_INPUT_1,
    REGISTER_OUTPUT_0,
    REGISTER_OUTPUT_1,
    REGISTER_OUTPUT_2,
    REGISTER_OUTPUT_3,
    REGISTER_NUM_ENUMS
} Register;

static const StackPtxRegister registers[] = {
    [REGISTER_INPUT_0] =    { .name = "input_register_0",   .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_INPUT_1] =    { .name = "input_register_1",   .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_OUTPUT_0] =   { .name = "output_register_0",  .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_OUTPUT_1] =   { .name = "output_register_1",  .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_OUTPUT_2] =   { .name = "output_register_2",  .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_OUTPUT_3] =   { .name = "output_register_3",  .stack_idx = STACK_PTX_STACK_TYPE_F32 },
};
static const size_t num_registers = STACK_PTX_ARRAY_NUM_ELEMS(registers);

// We need to add the names of the instructions so when an instruction is performed
// the compiler knows what to call it in ptx. Here the names come from the import
// "stack_ptx_generated_descriptions.h"

// Similarly we need to specify what to call an input register when the compiler
// processes input #0 and input #1.

// We describe a series of instructions. We push some U32 values on to the stack and
// perform some ptx operations on them. We push input_0 and input_1 on the stack as well 
// so we should see these registers show up as operands in the output ptx.
// stack_ptx_encode_return is mandatory at the end of the instructions list as a null terminator.
static const StackPtxInstruction instructions[] = {
    stack_ptx_encode_constant_f32(1.0f),
    stack_ptx_encode_constant_f32(2.0f),
    stack_ptx_encode_constant_f32(3.0f),
    stack_ptx_encode_constant_f32(4.0f),
    stack_ptx_encode_constant_f32(5.0f),
    stack_ptx_encode_constant_f32(6.0f),
    stack_ptx_encode_constant_f32(7.0f),
    stack_ptx_encode_ptx_instruction_cvt_rna_tf32_f32,
    stack_ptx_encode_ptx_instruction_cvt_rna_tf32_f32,
    stack_ptx_encode_ptx_instruction_cvt_rna_tf32_f32,
    stack_ptx_encode_ptx_instruction_mma_sync_aligned_m16n8k4_row_col_f32_tf32_tf32_f32,
    stack_ptx_encode_return
};

// We're going to request one U32 register value from what is on the U32 stack.
// The value on the U32 stack is an AST that will be evalutated to generate 
// the PTX code to give the result.

static const size_t requests[] = {
    REGISTER_OUTPUT_0, REGISTER_OUTPUT_1, REGISTER_OUTPUT_2, REGISTER_OUTPUT_3
};
static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

// We will allow the stack machine to run "execution_limit" number of operations before halting
static const size_t execution_limit = 100;

int
main() {
    // We first need to query the workspace necessary for the `stack_ptx_compile` function.
    // We set the max AST size to use during the run and 
    // the AST visit depth. Every operation is encoded in to the AST during the execution of the program
    // before compilation. AST visit depth is the max stack depth as we visit the AST during compilation.
    // We use the stackPtxCheck macro to error out on a non STACK_PTX_SUCCESS result.
    size_t stack_ptx_workspace_size;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &stack_ptx_workspace_size
        )
    );

    // We allocate the memory for the workspace.
    void* stack_ptx_workspace = malloc(stack_ptx_workspace_size);
    
    char* buffer = NULL;
    size_t required = 0;
    size_t capacity = 0;
    
    // We first "measure" the size of the output buffer by passing NULL as the output buffer.
    // This will give us the amount of bytes we require to write the output ptx.
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            requests, num_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            NULL,
            capacity,
            &required
        )
    );

    // Allocate the required buffer size, factoring in the string null terminator
    capacity = required + 1;
    buffer = (char*)malloc(capacity);

    // Now compile to ptx with the actual buffer.
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers,
            REGISTER_NUM_ENUMS,
            NULL,
            0,
            requests,
            num_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            buffer,
            capacity,
            &required
        )
    );

    // We don't need the workspace anymore.
    free(stack_ptx_workspace);

    // Print the ptx buffer
    printf("%s\n", buffer);

    // We don't need the buffer anymore.
    free(buffer);

    return 0;
}
