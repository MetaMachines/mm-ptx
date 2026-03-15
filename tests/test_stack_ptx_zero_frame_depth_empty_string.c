/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <check_result_helper.h>

#include <stdlib.h>

int
main(void) {
    static const char* stack_literal_prefixes[] = {
        "u32"
    };

    static const StackPtxArgTypeInfo arg_type_info[] = {
        { .stack_idx = 0, .num_vec_elems = 0 }
    };

    static const StackPtxStackInfo stack_info = {
        .ptx_instruction_strings = NULL,
        .ptx_instruction_descriptors = NULL,
        .num_ptx_instructions = 0,
        .special_register_strings = NULL,
        .special_register_descriptors = NULL,
        .num_special_registers = 0,
        .stack_literal_prefixes = stack_literal_prefixes,
        .num_stacks = 1,
        .arg_type_info = arg_type_info,
        .num_arg_types = 1
    };

    static const StackPtxCompilerInfo compiler_info = {
        .max_ast_size = 1,
        .max_ast_to_visit_stack_depth = 1,
        .stack_size = 1,
        .max_frame_depth = 0,
        .store_size = 0
    };

    size_t workspace_size = 0;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_info,
            &workspace_size
        )
    );

    void* workspace = malloc(workspace_size);
    ASSERT(workspace != NULL);

    char buffer[1] = { 'x' };
    size_t bytes_written = 123;

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_info,
            NULL,
            NULL,
            0,
            NULL,
            0,
            NULL,
            0,
            0,
            workspace,
            workspace_size,
            buffer,
            sizeof(buffer),
            &bytes_written
        )
    );

    ASSERT(bytes_written == 0);
    ASSERT(buffer[0] == '\0');

    free(workspace);

    return 0;
}
