/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_default_generated_types.h>

#define STACK_PTX_DEBUG
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_default_generated_types.h>
#include <stack_ptx_default_info.h>

#include <helpers.h>
#include <omp.h>
#include <nvJitLink.h>
#include <string.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

INCTXT(annotated_ptx, XSTRING(PTX_KERNEL));

int
main() {
    printf("%s\n", g_annotated_ptx_data);
    PtxInjectHandle ptx_inject;
    ptxInjectCheck(
        ptx_inject_create(
            &ptx_inject, 
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            g_annotated_ptx_data
        )
    );

    size_t num_injects;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects) );

    // All Inject sites should be the same as they're duplicated N times.
    enum Register {
        REGISTER_X,
        REGISTER_Y,
        REGISTER_Z,
        REGISTER_NUM_ENUMS
    };

    const char* cuda_variable_names[] = { 
        [REGISTER_X] = "x", 
        [REGISTER_Y] = "y", 
        [REGISTER_Z] = "z"
    };

    StackPtxRegister registers[REGISTER_NUM_ENUMS];
    static const size_t num_registers = REGISTER_NUM_ENUMS;

    static const size_t requests[] = {
        REGISTER_Z
    };
    static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

    size_t first_inject_idx = 0;
    for (size_t i = 0; i < num_registers; i++) {
        StackPtxRegister* reg = &registers[i];
        reg->stack_idx = STACK_PTX_STACK_TYPE_F32;
        const char* cuda_variable_name = cuda_variable_names[i];
        ptxInjectCheck( 
            ptx_inject_variable_info_by_name(ptx_inject, first_inject_idx, cuda_variable_name, NULL, NULL, NULL, &reg->name) 
        );
    }

    // Now register mappings are loaded.

    const char** ptx_stubs = (const char**)malloc(num_injects * sizeof(const char*));

    size_t stack_ptx_workspace_size;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &stack_ptx_workspace_size
        )
    );

    void* stack_ptx_workspace = malloc(stack_ptx_workspace_size);

    static const size_t execution_limit = 100;
    size_t capacity = 1000000ull;
    size_t required;

    char* buffer = (char*)malloc(capacity);
    char* buffer_ptr = buffer;
    for (size_t i = 0; i < num_injects; i++) {
        const StackPtxInstruction instructions[] = {
            stack_ptx_encode_input(REGISTER_X),
            stack_ptx_encode_input(REGISTER_Y),
            stack_ptx_encode_constant_f32((float)i),
            stack_ptx_encode_ptx_instruction_add_ftz_f32,
            stack_ptx_encode_ptx_instruction_add_ftz_f32,
            stack_ptx_encode_return
        };

        ptx_stubs[i] = buffer_ptr;
        stackPtxCheck(
            stack_ptx_compile(
                &compiler_info,
                &stack_ptx_stack_info,
                instructions,
                registers,
                num_registers,
                NULL, 0,
                requests,
                num_requests,
                execution_limit,
                stack_ptx_workspace,
                stack_ptx_workspace_size,
                buffer_ptr,
                capacity,
                &required
            )
        );
        buffer_ptr += required;
        capacity -= required;

        // NULL terminate and adjust accordingly
        *buffer_ptr = '\0';
        buffer_ptr++;
        capacity--;

    }

    for (size_t i = 0; i < num_injects; i++) {
        printf("inject #%zu:\n%s\n", i, ptx_stubs[i]);
    }
}
