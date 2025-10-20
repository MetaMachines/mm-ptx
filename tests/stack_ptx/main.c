/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define STACK_PTX_DEBUG
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_default_generated_types.h>
#include <stack_ptx_default_info.h>

#include <check_result_helper.h>

int
main() {
    static const size_t execution_limit = 100;

    size_t workspace_size;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &workspace_size
        )
    );

    void* workspace = malloc(workspace_size);

    typedef enum {
        REGISTER_INPUT_X,
        REGISTER_INPUT_Y,
        REGISTER_OUTPUT_Z,
        REGISTER_NUM_ENUMS
    } Register;

    static const StackPtxRegister registers[] = {
        [REGISTER_INPUT_X] = { .name = "in_x", .stack_idx = STACK_PTX_STACK_TYPE_U32},
        [REGISTER_INPUT_Y] = { .name = "in_y", .stack_idx = STACK_PTX_STACK_TYPE_U32},
        [REGISTER_OUTPUT_Z] ={ .name = "out_z", .stack_idx = STACK_PTX_STACK_TYPE_U32},
    };

    static const size_t request = REGISTER_OUTPUT_Z;

    static const StackPtxInstruction instructions[] = {
        stack_ptx_encode_input(REGISTER_INPUT_X),
        stack_ptx_encode_input(REGISTER_INPUT_Y),
        stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_U32),
        stack_ptx_encode_ptx_instruction_add_u32,
        stack_ptx_encode_return
    };

    static const size_t buffer_size = 1000000ull;
    char *buffer = (char*)malloc(buffer_size);
    size_t buffer_bytes_written;
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            &request, 1,
            execution_limit,
            workspace,
            workspace_size,
            buffer,
            buffer_size,
            &buffer_bytes_written
        )
    );

    printf("%s\n", buffer);

    free(buffer);
    free(workspace);
}
