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
    REGISTER_OUTPUT,
    REGISTER_NUM_ENUMS
} Register;

static const StackPtxRegister registers[] = {
    [REGISTER_INPUT_0] =   { .name = "input_register_0", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_INPUT_1] =   { .name = "input_register_1", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
    [REGISTER_OUTPUT] =    { .name = "output_register", .stack_idx = STACK_PTX_STACK_TYPE_U32 },
};

// We'll add a few special registers this time.
static const StackPtxInstruction instructions[] = {

    stack_ptx_encode_special_register_warpid,
    stack_ptx_encode_special_register_clock, // This will add %clock as an operand in PTX.
    stack_ptx_encode_special_register_tid, // This will add all four tid values to the stack as %tid.x, %tid.y etc..
    stack_ptx_encode_special_register_warpid, // Special register %warpid.
    stack_ptx_encode_special_register_warpid,
    stack_ptx_encode_input(REGISTER_INPUT_0),
    stack_ptx_encode_input(REGISTER_INPUT_1),
    stack_ptx_encode_constant_u32(1),
    stack_ptx_encode_ptx_instruction_mul_lo_u32,
    stack_ptx_encode_ptx_instruction_add_u32,   // A bunch of add instructions so everything will show up in output.
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_ptx_instruction_add_u32,
    stack_ptx_encode_return
};

// We're going to request one U32 register value from what is on the U32 stack.
// The value on the U32 stack is an AST that will be evalutated to generate 
// the PTX code to give the result.
static const size_t requests[] = { REGISTER_OUTPUT };
static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

// We will allow the stack machine to run "execution_limit" number of operations before halting
static const int execution_limit = 200;

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
            NULL, 0,
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
            NULL, 0,
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
