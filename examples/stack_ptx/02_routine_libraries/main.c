/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

// Import Stack PTX for this compilation using STACK_PTX_IMPLEMENTATION
#define STACK_PTX_DEBUG
// Can set the stack frame depth to allow deeper calls of routines to other routines (default: 3)
// #define STACK_PTX_FRAME_DEPTH 10
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_example_descriptions.h>
#include <stack_ptx_default_info.h>

// We need to bring in the #defines first to setup the order with
// RoutineIdx so the routines can reference these positions.
#include <routine_library_f32.h>
#include <routine_library_u32.h>

#include <check_result_helper.h>

// Describe the routines as an index, this will be useful later.
enum RoutineIdx {
    ROUTINE_LIBRARY_U32, // This brings in the enums that ROUTINE_LIBRARY_U32 needs
    ROUTINE_LIBRARY_F32, // This brings in the enums that ROUTINE_LIBRARY_F32 needs
    ROUTINE_0,
    ROUTINE_1,
    ROUTINE_2,
    NUM_ROUTINES
};

// Now we can bring in the actual routines which now have the right indices for
// Where other routines they might reference are.
#include <routine_library_f32_impl.h>
#include <routine_library_u32_impl.h>

// Describe the routines, use designated initializers to position
// them properly in the array for indexing.
// We need to add the #defines for the initializers here to bring
// in the actual routines
static const StackPtxInstruction* routines[] = {
    ROUTINE_LIBRARY_U32_INITIALIZERS, // Add this in the same order as added in RoutineIdx for ROUTINE_LIBRARY_U32
    ROUTINE_LIBRARY_F32_INITIALIZERS, // Add this in the same order as added in RoutineIdx for ROUTINE_LIBRARY_F32
    [ROUTINE_0] = (StackPtxInstruction[]){
        stack_ptx_encode_constant_u32(1023),
        stack_ptx_encode_return
    },
    [ROUTINE_1] = (StackPtxInstruction[]){
        stack_ptx_encode_constant_u32(1024),
        stack_ptx_encode_return
    },
    // ROUTINE_2 will call ROUTINE_1
    [ROUTINE_2] = (StackPtxInstruction[]){
        stack_ptx_encode_constant_u32(1025),
        stack_ptx_encode_routine(ROUTINE_1),
        stack_ptx_encode_return
    }
};

// Set up the register names idx.
typedef enum {
    REGISTER_INPUT_0,
    REGISTER_INPUT_1,
    REGISTER_OUTPUT,
    REGISTER_NUM_ENUMS
} Registers;

static const StackPtxRegister registers[] = {
    [REGISTER_INPUT_0] = { .name = "input_register_0", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_INPUT_1] = { .name = "input_register_1", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_OUTPUT] =  { .name = "output_register", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
};

// Set up the instructions we'll run
static const StackPtxInstruction instructions[] = {
    stack_ptx_encode_constant_u32(1),
    stack_ptx_encode_constant_u32(2),
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_constant_u32(3),
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_input(REGISTER_INPUT_0),
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_input(REGISTER_INPUT_1),
    stack_ptx_encode_routine(ROUTINE_2),
    stack_ptx_encode_routine(ROUTINE_LIBRARY_U32_FUNC_0),
    stack_ptx_encode_ptx_instruction_mul_lo_u32,
    stack_ptx_encode_return
};

// We're going to request one U32 register value from the ast
static const size_t requests[] = { REGISTER_OUTPUT };
static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

// Set execution limit for running instructions
static const int execution_limit = 100;

int
main() {
    size_t stack_ptx_workspace_size;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info, 
            &stack_ptx_workspace_size
        )
    );

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
            routines, NUM_ROUTINES,
            requests,
            num_requests,
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
            registers, REGISTER_NUM_ENUMS,
            routines, NUM_ROUTINES,
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

    free(stack_ptx_workspace);

    // Print the ptx buffer
    printf("%s\n", buffer);

    free(buffer);

    return 0;
}
