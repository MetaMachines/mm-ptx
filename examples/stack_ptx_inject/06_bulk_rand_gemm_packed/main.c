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

#include <check_result_helper.h>
#include <cuda_helper.h>
#include <nvptx_helper.h>

#include <omp.h>
#include <nvJitLink.h>
#include <string.h>

#include <errno.h>
#include <inttypes.h>  // for strtoumax, uintmax_t
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include <mma_helper.h>

#include <cuda.h>

#include <ptx_inject_helper.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

INCTXT(annotated_ptx, XSTRING(PTX_KERNEL));

// Prints the InstructionLayout and the related matrix
#define MATRIX_PRINT_RESULTS            1
// Number of rows and columns to print of the matrix
#define MATRIX_PRINT_LIMIT              5

static
inline
int 
parse_size_t(
    const char *s, 
    size_t *out
) {
    if (!s || !*s) return EINVAL;

    errno = 0;
    char *end = NULL;

    // base 10; use base 0 if you want 0x... and 0... prefixes auto-detected
    uintmax_t v = strtoumax(s, &end, 10);

    if (errno == ERANGE) return ERANGE;          // overflow in uintmax_t
    if (end == s) return EINVAL;                 // no digits
    // reject trailing junk (allow trailing spaces if you like: skip them first)
    while (*end == ' ' || *end == '\t' || *end == '\n') end++;
    if (*end != '\0') return EINVAL;

    if (v > (uintmax_t)SIZE_MAX) return ERANGE;  // doesn't fit in size_t

    *out = (size_t)v;
    return 0;
}

static
inline
bool 
str_starts_with(
    const char *s, 
    const char *prefix
) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

typedef struct {
    StackPtxInstruction multiply_unary_v_a;
    StackPtxInstruction multiply_unary_v_b;
    StackPtxInstruction multiply_binary_v_a_v_b;
    StackPtxInstruction accumulate_binary_diff_v_c;
    StackPtxInstruction epilogue_unary_v_c;
} InstructionLayout;

__attribute__((unused))
static
void
print_instruction_layout(
    InstructionLayout instruction_layout
) {
    const char* multiply_unary_v_a_name = stack_ptx_ptx_instruction_display_names[instruction_layout.multiply_unary_v_a.idx];
    const char* multiply_unary_v_b_name = stack_ptx_ptx_instruction_display_names[instruction_layout.multiply_unary_v_b.idx];
    const char* multiply_binary_v_a_v_b_name = stack_ptx_ptx_instruction_display_names[instruction_layout.multiply_binary_v_a_v_b.idx];
    const char* accumulate_binary_diff_v_c_name = stack_ptx_ptx_instruction_display_names[instruction_layout.accumulate_binary_diff_v_c.idx];
    const char* epilogue_unary_v_c_name = stack_ptx_ptx_instruction_display_names[instruction_layout.epilogue_unary_v_c.idx];
    printf(
        "\tdiff = %s(%s(v_a), %s(v_b))\n",
        multiply_binary_v_a_v_b_name,
        multiply_unary_v_a_name,
        multiply_unary_v_b_name
    );
    printf(
        "\tv_c_out = %s(diff, v_c)\n",
        accumulate_binary_diff_v_c_name
    );
    printf(
        "\tv_c = %s(v_c)\n",
        epilogue_unary_v_c_name
    );
}

static
void
populate_instruction_layouts(
    InstructionLayout* instruction_layouts,
    size_t num_instruction_layouts
) {
    // Array of only the unary f32 PTX instructions we found.
    StackPtxInstruction unary_f32_instructions[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS];
    int num_unary_f32_instructions = 0;

    // Array of only the binary f32 PTX instructions we found.
    StackPtxInstruction binary_f32_instructions[STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS];
    int num_binary_f32_instructions = 0;

    // Gather the instructions.
    for (int i = 0; i < STACK_PTX_PTX_INSTRUCTION_NUM_ENUMS; i++) {
        StackPtxInstruction instruction = stack_ptx_ptx_instructions[i];
        StackPtxPTXArgs ptx_args = instruction.payload.ptx_args;

        // PTX Instruction needs to only return one F32 value.
        if (ptx_args.ret_0 != STACK_PTX_ARG_TYPE_F32) continue;
        if (ptx_args.ret_1 != STACK_PTX_ARG_TYPE_NONE) continue;

        // Needs to have first argument as a F32.
        // Unary has None as it's second argument.
        if (ptx_args.arg_0 != STACK_PTX_ARG_TYPE_F32) continue;
        if (ptx_args.arg_1 == STACK_PTX_ARG_TYPE_NONE) {
            unary_f32_instructions[num_unary_f32_instructions++] = instruction;
            continue;
        }
        // Binary needs to have second argument as a F32.
        // Binary has None as it's third argument.
        if (ptx_args.arg_1 != STACK_PTX_ARG_TYPE_F32) continue;
        if (ptx_args.arg_2 == STACK_PTX_ARG_TYPE_NONE) {
            binary_f32_instructions[num_binary_f32_instructions++] = instruction;
            continue;
        }
    }

    printf("Found %d unary f32 PTX instructions:\n", num_unary_f32_instructions);
    for (int i = 0; i < num_unary_f32_instructions; i++) {
        StackPtxInstruction instruction = unary_f32_instructions[i];
        printf("\t%s\n", stack_ptx_ptx_instruction_ptx_names[instruction.idx]);
    }

    printf("Found %d binary f32 PTX instructions:\n", num_binary_f32_instructions);
    for (int i = 0; i < num_binary_f32_instructions; i++) {
        StackPtxInstruction instruction = binary_f32_instructions[i];
        printf("\t%s\n", stack_ptx_ptx_instruction_ptx_names[instruction.idx]);
    }
    printf("\n");

    for (int i = 0; i < num_instruction_layouts; i++) {
        InstructionLayout* instruction_layout = &instruction_layouts[i];
        instruction_layout->multiply_unary_v_a =            unary_f32_instructions[rand() % num_unary_f32_instructions];
        instruction_layout->multiply_unary_v_b =            unary_f32_instructions[rand() % num_unary_f32_instructions];
        instruction_layout->multiply_binary_v_a_v_b =       binary_f32_instructions[rand() % num_binary_f32_instructions];
        instruction_layout->accumulate_binary_diff_v_c =    binary_f32_instructions[rand() % num_binary_f32_instructions];
        instruction_layout->epilogue_unary_v_c =            unary_f32_instructions[rand() % num_unary_f32_instructions];
    }
}

// All Inject sites should be the same as they're duplicated N times.
enum Register {
    REGISTER_MULTIPLY_OUTPUT_DIFF,
    REGISTER_MULTIPLY_INPUT_V_A,
    REGISTER_MULTIPLY_INPUT_V_B,
    REGISTER_ACCUMULATE_OUTPUT_V_C_OUT,
    REGISTER_ACCUMULATE_INPUT_V_C,
    REGISTER_ACCUMULATE_INPUT_DIFF,
    REGISTER_EPILOGUE_MODIFY_V_C,
    REGISTER_NUM_ENUMS
};

static const char* cuda_variable_names[] = { 
    [REGISTER_MULTIPLY_OUTPUT_DIFF] = "diff", 
    [REGISTER_MULTIPLY_INPUT_V_A] = "v_a", 
    [REGISTER_MULTIPLY_INPUT_V_B] = "v_b",
    [REGISTER_ACCUMULATE_OUTPUT_V_C_OUT] = "v_c_out",
    [REGISTER_ACCUMULATE_INPUT_V_C] = "v_c",
    [REGISTER_ACCUMULATE_INPUT_DIFF] = "diff",
    [REGISTER_EPILOGUE_MODIFY_V_C] = "v_c"
};

static const size_t num_registers = REGISTER_NUM_ENUMS;

// This definition comes from CMake
static const size_t num_kernels = NUM_KERNEL_REPLICATIONS;

int
main() {
    InstructionLayout* instruction_layouts = (InstructionLayout*)malloc(num_kernels * sizeof(InstructionLayout)); 
    populate_instruction_layouts(instruction_layouts, num_kernels);

    for (size_t i = 0; i < num_kernels; i++) {
        InstructionLayout instruction_layout = instruction_layouts[i];
        print_instruction_layout(instruction_layout);
    }

    StackPtxRegister registers[REGISTER_NUM_ENUMS];
    for (size_t i = 0; i < REGISTER_NUM_ENUMS; i++) {
        registers[i].stack_idx = STACK_PTX_STACK_TYPE_F32;
    }

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

    print_ptx_inject_info(
        ptx_inject,
        ptx_inject_data_type_infos
    );

    static const size_t num_injects_per_func = 3;
    size_t num_injects;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects) );
    size_t num_funcs = num_injects / num_injects_per_func;
    ASSERT( num_injects % num_injects_per_func == 0 );

    size_t multiply_func_idx;
    size_t accumulate_func_idx;
    size_t epilogue_func_idx;

    size_t multiply_num_args;
    size_t multiply_num_sites;
    ptxInjectCheck( 
        ptx_inject_inject_info_by_name(ptx_inject, "multiply000000", &multiply_func_idx, &multiply_num_args, &multiply_num_sites)
    );

    size_t accumulate_num_args;
    size_t accumulate_num_sites;
    ptxInjectCheck( 
        ptx_inject_inject_info_by_name(ptx_inject, "accumulate000000", &accumulate_func_idx, &accumulate_num_args, &accumulate_num_sites)
    );

    size_t epilogue_num_args;
    size_t epilogue_num_sites;
    ptxInjectCheck( 
        ptx_inject_inject_info_by_name(ptx_inject, "epilogue000000", &epilogue_func_idx, &epilogue_num_args, &epilogue_num_sites) 
    );
    
    // Another sanity check for the kernel.
    ASSERT( multiply_num_args == 3 );
    ASSERT( accumulate_num_args == 3 );
    ASSERT( epilogue_num_args == 1 );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "diff", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_OUTPUT_DIFF].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "v_a", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_INPUT_V_A].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "v_b", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_INPUT_V_B].name) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "v_c_out", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_OUTPUT_V_C_OUT].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "v_c", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_INPUT_V_C].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "diff", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_INPUT_DIFF].name) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, epilogue_func_idx, "v_c", NULL, NULL, NULL, &registers[REGISTER_EPILOGUE_MODIFY_V_C].name) );

    static const size_t multiply_requests[] = { REGISTER_MULTIPLY_OUTPUT_DIFF };
    static const size_t num_multiply_requests = STACK_PTX_ARRAY_NUM_ELEMS(multiply_requests);

    static const size_t accumulate_requests[] = { REGISTER_ACCUMULATE_OUTPUT_V_C_OUT };
    static const size_t num_accumulate_requests = STACK_PTX_ARRAY_NUM_ELEMS(accumulate_requests);

    static const size_t epilogue_requests[] = { REGISTER_EPILOGUE_MODIFY_V_C };
    static const size_t num_epilogue_requests = STACK_PTX_ARRAY_NUM_ELEMS(epilogue_requests);

    typedef struct {
        size_t mulitply_func_idx;
        size_t accumulate_func_idx;
        size_t epilogue_func_idx;
    } FuncIndices;

    // We're going to store it [num_funcs][3]
    size_t* reverse_indices = (size_t*)malloc(num_injects * sizeof(size_t));

    typedef enum {
        IT_MULTIPLY,
        IT_ACCUMULATE,
        IT_EPILOGUE,
        IT_NUM_ENUMS
    } InjectType;

    // This will figure out where each program goes in the stub array.
    for (size_t i = 0; i < num_injects; i++) {
        const char* inject_name;
        ptxInjectCheck(
            ptx_inject_inject_info_by_index(
                ptx_inject,
                i,
                &inject_name,
                NULL,
                NULL
            )
        );

        InjectType inject_type;
        if (str_starts_with(inject_name, "multiply")) {
            inject_name += strlen("multiply");
            inject_type = IT_MULTIPLY;
        } else if (str_starts_with(inject_name, "accumulate")) {
            inject_name += strlen("accumulate");
            inject_type = IT_ACCUMULATE;
        } else if (str_starts_with(inject_name, "epilogue")) {
            inject_name += strlen("epilogue");
            inject_type = IT_EPILOGUE;
        } else {
            ASSERT( false );
        }

        size_t kernel_idx;
        int ret = parse_size_t(inject_name, &kernel_idx);
        ASSERT( ret == 0 );

        reverse_indices[i] = kernel_idx * IT_NUM_ENUMS + inject_type;
    }

    // Now in this loop we can create our "programs" and set the buffer pointer for the ptx_stubs
    const char** ptx_stubs = (const char**)malloc(num_injects * sizeof(const char*));
    size_t buffer_size = 100000000ull;
    char* buffer = (char*)malloc(buffer_size);
    size_t buffer_capacity = buffer_size;
    char* buffer_ptr = buffer;
    size_t bytes_written;
    size_t workspace_size;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &workspace_size
        )
    );

    char* workspace = buffer_ptr;
    buffer_ptr += workspace_size;
    buffer_capacity -= workspace_size;

    // This is going to be traversed in the order of the programs
    for (size_t i = 0; i < num_funcs; i++) {
        printf("%zu: %zu\n", i, reverse_indices[i]);
        size_t inject_idx_multiply = i * IT_NUM_ENUMS + IT_MULTIPLY;
        size_t inject_idx_accumulate = i * IT_NUM_ENUMS + IT_ACCUMULATE;
        size_t inject_idx_epilogue = i * IT_NUM_ENUMS + IT_EPILOGUE;

        InstructionLayout* instruction_layout = &instruction_layouts[i];
        const StackPtxInstruction multiply_instructions[] = {
            stack_ptx_encode_input(REGISTER_MULTIPLY_INPUT_V_A),
            instruction_layout->multiply_unary_v_a,
            stack_ptx_encode_input(REGISTER_MULTIPLY_INPUT_V_B),
            instruction_layout->multiply_unary_v_b,
            instruction_layout->multiply_binary_v_a_v_b,
            stack_ptx_encode_return
        };

        const StackPtxInstruction accumulate_instructions[] = {
            stack_ptx_encode_input(REGISTER_ACCUMULATE_INPUT_V_C),
            stack_ptx_encode_input(REGISTER_ACCUMULATE_INPUT_DIFF),
            instruction_layout->accumulate_binary_diff_v_c,
            stack_ptx_encode_return
        };

        const StackPtxInstruction epilogue_instructions[] = {
            stack_ptx_encode_input(REGISTER_EPILOGUE_MODIFY_V_C),
            instruction_layout->epilogue_unary_v_c,
            stack_ptx_encode_return
        };

        
        stackPtxCheck(
            stack_ptx_compile(
                &compiler_info,
                &stack_ptx_stack_info,
                multiply_instructions,
                registers, num_registers,
                NULL, 0,
                multiply_requests, num_multiply_requests,
                100,
                workspace, workspace_size,
                buffer_ptr, buffer_capacity,
                &bytes_written
            )
        );
        
        ptx_stubs[reverse_indices[inject_idx_multiply]] = buffer_ptr;
        buffer_ptr += bytes_written + 1;
        buffer_capacity -= bytes_written + 1; 

        stackPtxCheck(
            stack_ptx_compile(
                &compiler_info,
                &stack_ptx_stack_info,
                accumulate_instructions,
                registers, num_registers,
                NULL, 0,
                accumulate_requests, num_accumulate_requests,
                100,
                workspace, workspace_size,
                buffer_ptr, buffer_capacity,
                &bytes_written
            )
        );
        
        ptx_stubs[reverse_indices[inject_idx_accumulate]] = buffer_ptr;
        buffer_ptr += bytes_written + 1;
        buffer_capacity -= bytes_written + 1; 

        stackPtxCheck(
            stack_ptx_compile(
                &compiler_info,
                &stack_ptx_stack_info,
                epilogue_instructions,
                registers, num_registers,
                NULL, 0,
                epilogue_requests, num_epilogue_requests,
                100,
                workspace, workspace_size,
                buffer_ptr, buffer_capacity,
                &bytes_written
            )
        );
        
        ptx_stubs[reverse_indices[inject_idx_epilogue]] = buffer_ptr;
        buffer_ptr += bytes_written + 1;
        buffer_capacity -= bytes_written + 1; 
        // ptx_stubs[reverse_indices[inject_idx_accumulate]] = 
        // ptx_stubs[reverse_indices[inject_idx_epilogue]] = 
    }

    for(size_t i = 0; i < num_injects; i++) {
        const char* ptx_stub = ptx_stubs[i];
        printf("(%zu): %s\n", i, ptx_stub);
    }

    PtxInjectResult ptx_inject_result = 
        ptx_inject_render_ptx(
            ptx_inject, ptx_stubs,
            num_injects,
            buffer_ptr, 
            buffer_capacity,
            &bytes_written
        );
    
    if (ptx_inject_result != PTX_INJECT_SUCCESS) {
        if (ptx_inject_result == PTX_INJECT_ERROR_INSUFFICIENT_BUFFER) {
            printf(
                "Insufficient buffer for PTX Inject: Requested(%zu), Got(%zu)\n",
                bytes_written, buffer_capacity
            );
        } 
        ptxInjectCheck(ptx_inject_result);
    }

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    
    cuCheck( cuDeviceGet(&cu_device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0, sm_%d%d)\n\n", device_compute_capability_major, device_compute_capability_minor);

    // Put together the nvPTX compiler options with the proper architecture.
    char nv_ptx_compile_line_buffer[32];
    sprintf(nv_ptx_compile_line_buffer, "--gpu-name=sm_%d%d", device_compute_capability_major, device_compute_capability_minor);

    const char* ptx_compile_options[] = {
        nv_ptx_compile_line_buffer,
        // "--compile-only"
        // "--Ofast-compile=max"
    };
    const size_t num_ptx_compile_options = 1;

    time_t start, end;
    double elapsed;

    time(&start);
    nvPTXCompilerHandle nvptx_compiler;
    nvptxCheck(
        nvPTXCompilerCreate(
            &nvptx_compiler,
            bytes_written,
            buffer_ptr
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

    time(&end);  // End timing
    elapsed = difftime(end, start);  // Difference in seconds

    printf("Elapsed time: %.0f seconds\n", elapsed);

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

    ASSERT( function_count == num_funcs );
    CUfunction* functions = (CUfunction*)malloc(function_count * sizeof(CUfunction));

    cuCheck( cuModuleEnumerateFunctions(functions, function_count, module) );

    for (size_t i = 0; i < function_count; i++) {
        const char* func_name;
        cuCheck( cuFuncGetName(&func_name, functions[i]) );
        printf("name: %s\n", func_name);
    }

    static const int M = 128;
    static const int N = 128;
    static const int K = 8;
    static const int lda = 128;
    static const int ldb = 128;
    static const int ldc = 128;
    static const unsigned int block_dim = 256;

    float* h_a = (float*)malloc((size_t)M * (size_t)K * sizeof(float));
    float* h_b = (float*)malloc((size_t)N * (size_t)K * sizeof(float));
    float* h_c = (float*)malloc((size_t)M * (size_t)N * sizeof(float));

    for (int j = 0; j < M*K; ++j) h_a[j] = 2*(rand() / (double)(RAND_MAX)) - 1;
    for (int j = 0; j < N*K; ++j) h_b[j] = 2*(rand() / (double)(RAND_MAX)) - 1;

    CUdeviceptr d_a;
    CUdeviceptr d_b;
    CUdeviceptr d_cs;

    size_t d_c_matrix_size = (size_t)M * (size_t)N * sizeof(float);

    cuCheck( cuMemAlloc(&d_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_b, (size_t)N * (size_t)K * sizeof(float)) );
    // Allocate an output tensor for each kernel.
    cuCheck( cuMemAlloc(&d_cs, d_c_matrix_size * num_kernels) );

    cuCheck( cuMemcpyHtoD(d_a, h_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemcpyHtoD(d_b, h_b, (size_t)N * (size_t)K * sizeof(float)) );

    CUstream stream;
    cuCheck( cuStreamCreate(&stream, 0) );

    double time_start, time_end;
    
    time_start = omp_get_wtime();
    for (int i = 0; i < num_kernels; i++) {
        CUdeviceptr d_c = d_cs + (size_t)i * d_c_matrix_size;
        void* args[] = {
            (void*)&M,
            (void*)&N,
            (void*)&K,
            (void*)&d_a,
            (void*)&lda,
            (void*)&d_b,
            (void*)&ldb,
            (void*)&d_c,
            (void*)&ldc
        };

        cuCheck( 
            cuLaunchKernel(
                functions[i],
                1, 1, 1,
                block_dim, 1, 1,
                0, 0, 
                args,
                NULL
            )
        );
    }
    cuCheck( cuCtxSynchronize() );
    time_end = omp_get_wtime();
    printf(
        "Ran Random Gemm Kernels (%d kernels, 1 cuda stream, %f seconds)\n",
        num_kernels,
        (time_end - time_start)
    );

    if (MATRIX_PRINT_RESULTS) {
        for (int i = 0; i < num_kernels; i++) {
            CUdeviceptr d_c = d_cs + (size_t)i * d_c_matrix_size;
            cuCheck( cuMemcpyDtoH(h_c, d_c, d_c_matrix_size) );
            const char* name;
            cuCheck( cuFuncGetName(&name, functions[i]));
            printf("Kernel (%s):\n", name);
            print_instruction_layout(instruction_layouts[i]);
            printf("------------------------\n");
            print_matrix(M, N, h_c, ldc, MATRIX_PRINT_LIMIT);
            printf("------------------------\n");
        }
    }
    
    cuCheck( cuMemFree(d_a) );
    cuCheck( cuMemFree(d_b) );
    cuCheck( cuMemFree(d_cs) );

    free(h_a);
    free(h_b);
    free(h_c);

    free(functions);
    cuCheck( cuModuleUnload(module) );
    cuCheck( cuStreamDestroy(stream) );
    cuCheck( cuCtxDestroy(cu_context) );
}
