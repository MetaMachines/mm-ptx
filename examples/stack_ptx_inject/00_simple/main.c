/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_default_generated_types.h>

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_default_generated_types.h>
#include <stack_ptx_default_info.h>

#include <check_result_helper.h>
#include <cuda.h>
#include <helpers.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

/* Use incbin to bring the code from kernel.ptx, allows easy editing of cuda source
*   is replaced with g_annotated_ptx_data
*/
INCBIN(char, annotated_ptx, XSTRING(PTX_KERNEL));

static const int execution_limit = 100;

static
float
run_stack_ptx_instructions(
    int device_compute_capability_major,
    int device_compute_capability_minor,
    CUdeviceptr d_out,
    PtxInjectHandle ptx_inject,
    const StackPtxRegister* registers,
    size_t num_registers,
    const size_t* requests,
    size_t num_requests,
    const StackPtxInstruction* instructions,
    void* workspace,
    size_t workspace_size
) {
    size_t required = 0;
    size_t capacity = 0;
    
    // Get the size of the buffer that will store the generated ptx
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
            workspace,
            workspace_size,
            NULL,
            capacity,
            &required
        )
    );

    // Account for null terminator
    capacity = required + 1;
    char* stub_buffer = (char *)malloc(capacity);

    // Run again to store the generated ptx to the buffer
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
            workspace,
            workspace_size,
            stub_buffer,
            capacity,
            &required
        )
    );

    // If there are more than one injects in the ptx, then we need to know which order to send the
    // stubs to 'ptx_inject_render_ptx. In this case because there is only one, 'func' will
    // always be at index 0.

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    const char* ptx_stubs[1];
    ptx_stubs[inject_func_idx] = stub_buffer;

    // We'll use the local helper this time
    size_t num_bytes_written;
    char* rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 1, &num_bytes_written);
    rendered_ptx[num_bytes_written-1] = '\0';

    // We should now see the add instruction inside the ptx.
    printf(
        "---------------------------------------------\n"
        "%s"
        "---------------------------------------------\n",
        rendered_ptx
    );
    
    // We can now compile this ptx to sass
    void* sass = nvptx_compile(device_compute_capability_major, device_compute_capability_minor, rendered_ptx, num_bytes_written, false);

    // Free rendered_ptx buffer
    free(rendered_ptx);

    // Now let's run this kernel!

    CUmodule cu_module;
    cuCheck( cuModuleLoadDataEx(&cu_module, sass, 0, NULL, NULL) );
    // We can free the sass
    free(sass);

    CUfunction cu_function;
    // Because we added 'extern "C"' "kernel" works as a name unmangled
    cuCheck( cuModuleGetFunction(&cu_function, cu_module, "kernel") );

    void* args[] = {
        (void*)&d_out
    };

    cuCheck( 
        cuLaunchKernel(
            cu_function,
            1, 1, 1,
            1, 1, 1,
            0, 0, 
            args,
            NULL
        )
    );

    cuCheck( cuCtxSynchronize() );
    cuCheck( cuModuleUnload(cu_module) );

    float h_out;
    cuCheck( cuMemcpyDtoH(&h_out, d_out, sizeof(float)) );

    return h_out;
}

int
main() {
    printf("\nAnnotated PTX\n"
        "---------------------------------------------\n"
        "%.*s"
        "---------------------------------------------\n\n",
         g_annotated_ptx_size, g_annotated_ptx_data
    );

    // The cmake plumbing already used the ptxinject cli tool compiled inside the 
    // project to process kernel.cu. The cuda was then compiled by nvcc as part of
    // the cmake process as well. INCBIN added the ptx to this file as g_annotated_ptx_data.

    PtxInjectHandle ptx_inject;
    ptxInjectCheck(
        ptx_inject_create(
            &ptx_inject, 
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            g_annotated_ptx_data
        )
    );

    enum Register {
        REGISTER_X,
        REGISTER_Y,
        REGISTER_Z,
        REGISTER_NUM_ENUMS
    };

    StackPtxRegister registers[] = {
        [REGISTER_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
        [REGISTER_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
        [REGISTER_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32},
    };
    static const size_t num_registers = REGISTER_NUM_ENUMS;

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "x", NULL, NULL, NULL, &registers[REGISTER_X].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "y", NULL, NULL, NULL, &registers[REGISTER_Y].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "z", NULL, NULL, NULL, &registers[REGISTER_Z].name) );

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

    static const size_t requests[] = {
        REGISTER_Z
    };
    static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

    static const StackPtxInstruction add_inputs[] = {
        stack_ptx_encode_input(REGISTER_X),
        stack_ptx_encode_input(REGISTER_Y),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction mul_inputs[] = {
        stack_ptx_encode_input(REGISTER_X),
        stack_ptx_encode_input(REGISTER_Y),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction mishmash_inputs[] = {
        stack_ptx_encode_input(REGISTER_X),
        stack_ptx_encode_ptx_instruction_sin_approx_ftz_f32,
        stack_ptx_encode_input(REGISTER_Y),
        stack_ptx_encode_ptx_instruction_cos_approx_ftz_f32,
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_constant_f32(1.4f),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    
    cuCheck( cuDeviceGet(&cu_device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, cu_device) );

    CUdeviceptr d_out;
    cuCheck( cuMemAlloc(&d_out, sizeof(float)) );

    printf("Inject (add_inputs) PTX\n");

    float add_result = 
        run_stack_ptx_instructions(
            device_compute_capability_major,
            device_compute_capability_minor,
            d_out,
            ptx_inject,
            registers, num_registers,
            requests,
            num_requests,
            add_inputs,
            stack_ptx_workspace,
            stack_ptx_workspace_size
        );

    printf("Inject (mul_inputs) PTX\n");

    float mul_result = 
        run_stack_ptx_instructions(
            device_compute_capability_major,
            device_compute_capability_minor,
            d_out,
            ptx_inject,
            registers, num_registers,
            requests,
            num_requests,
            mul_inputs,
            stack_ptx_workspace,
            stack_ptx_workspace_size
        );

    printf("Inject (mishmash_inputs) PTX\n");

    float mishmash_result = 
        run_stack_ptx_instructions(
            device_compute_capability_major,
            device_compute_capability_minor,
            d_out,
            ptx_inject,
            registers, num_registers,
            requests,
            num_requests,
            mishmash_inputs,
            stack_ptx_workspace,
            stack_ptx_workspace_size
        );

    cuCheck( cuMemFree(d_out) );
    cuCheck( cuCtxDestroy(cu_context) );

    free(stack_ptx_workspace);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    printf("Inject (add_inputs) result (should be 8):\t%f\n", add_result);
    printf("Inject (mul_inputs) result (should be 15):\t%f\n", mul_result);
    printf("Inject (mishmash_inputs) result:\t\t%f\n", mishmash_result);
}
