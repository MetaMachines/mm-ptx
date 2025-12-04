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

#include <ptx_inject_helper.h>
#include <cuda_helper.h>
#include <nvptx_helper.h>

#include <omp.h>
#include <nvJitLink.h>
#include <string.h>
#include <cuda.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

INCTXT(annotated_ptx, PTX_KERNEL);

int
main() {
    // printf("%s\n", g_annotated_ptx_data);
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

    print_ptx_inject_info(
        ptx_inject,
        ptx_inject_data_type_infos
    );

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

        // Adjust for NULL terminator
        buffer_ptr++;
        capacity--;

    }

    // for (size_t i = 0; i < num_injects; i++) {
    //     printf("inject #%zu:\n%s\n", i, ptx_stubs[i]);
    // }

    char* rendered_ptx = buffer_ptr;
    ptxInjectCheck(
        ptx_inject_render_ptx(
            ptx_inject,
            ptx_stubs,
            num_injects,
            buffer_ptr,
            capacity,
            &required
        )
    );
    buffer_ptr += required;
    capacity -= required;

    // Adjust for NULL terminator
    buffer_ptr++;
    capacity--;

    // printf(
    //     "---------------------\n"
    //     "%s\n"
    //     "---------------------\n", 
    //     rendered_ptx
    // );

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    
    cuCheck( cuDeviceGet(&cu_device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    // printf("Device(0, sm_%d%d)\n\n", device_compute_capability_major, device_compute_capability_minor);

    // Put together the nvPTX compiler options with the proper architecture.
    char nv_ptx_compile_line_buffer[32];
    sprintf(nv_ptx_compile_line_buffer, "--gpu-name=sm_%d%d", device_compute_capability_major, device_compute_capability_minor);

    const char* ptx_compile_options[] = {
        nv_ptx_compile_line_buffer,
        "--compile-only"
    };
    const size_t num_ptx_compile_options = 2;

    nvPTXCompilerHandle nvptx_compiler;
    nvptxCheck(
        nvPTXCompilerCreate(
            &nvptx_compiler,
            required,
            rendered_ptx
        )
    );
        
    nvPTXCompileResult result =
        nvPTXCompilerCompile(
            nvptx_compiler,
            num_ptx_compile_options,
            ptx_compile_options
        );

    if (result != NVPTXCOMPILE_SUCCESS) {
        nvptx_print_error_log(nvptx_compiler);
        assert( false );
        exit(1);
    }

    size_t sass_image_size;
    nvptxCheck( 
        nvPTXCompilerGetCompiledProgramSize(
            nvptx_compiler, 
            &sass_image_size
        )
    );
    void* sass_image = malloc(sass_image_size);

    nvptxCheck(
        nvPTXCompilerGetCompiledProgram(
            nvptx_compiler, 
            sass_image
        )
    );

    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, cu_device) );

    CUmodule module;
    cuCheck( cuModuleLoadDataEx(&module, sass_image, 0, NULL, NULL) );

    unsigned int function_count;
    cuCheck( cuModuleGetFunctionCount(&function_count, module) );

    ASSERT( function_count == num_injects );
    CUfunction* functions = (CUfunction*)malloc(function_count * sizeof(CUfunction));

    cuCheck( cuModuleEnumerateFunctions(functions, function_count, module) );

    // for (size_t i = 0; i < function_count; i++) {
    //     const char* func_name;
    //     cuCheck( cuFuncGetName(&func_name, functions[i]) );
    //     printf("name: %s\n", func_name);
    // }

    free(sass_image);
    free(ptx_stubs);
    free(stack_ptx_workspace);
    free(buffer);

    CUdeviceptr d_outs;
    cuCheck( cuMemAlloc(&d_outs, function_count * sizeof(float)) );

    for (size_t i = 0; i < function_count; i++) {
        CUdeviceptr d_out = d_outs + i * sizeof(float);
        CUfunction cu_function = functions[i];

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
    }

    cuCheck( cuCtxSynchronize() );
    cuCheck( cuModuleUnload(module) );

    float* h_outs = (float*)malloc(function_count * sizeof(float));
    cuCheck( cuMemcpyDtoH(h_outs, d_outs, function_count * sizeof(float)) );

    for (size_t i = 0; i < function_count; i++) {
        ASSERT( h_outs[i] == (float)(i + 8) );
    }

    printf("OK\n");

    free(h_outs);
    cuCheck( cuMemFree(d_outs) );
}
