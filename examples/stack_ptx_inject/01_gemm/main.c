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

#include <helpers.h>
#include <time.h>

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

#define STUB_BUFFER_SIZE 1000000ull

#define CLOCK_MULTIPLIER 1000000.0

static const int execution_limit = 100;

static
void
run_custom_mma(
    int M, int N, int K,
    CUdeviceptr d_a, int lda,
    CUdeviceptr d_b, int ldb,
    CUdeviceptr d_c, int ldc,
    int device_compute_capability_major,
    int device_compute_capability_minor,
    const char* rendered_ptx,
    size_t rendered_ptx_num_bytes
) {
    CUmodule cu_module;
    CUfunction cu_function;

    double time_start = clock();
    void* sass = 
        nvptx_compile(
            device_compute_capability_major, 
            device_compute_capability_minor, 
            rendered_ptx, 
            rendered_ptx_num_bytes, 
            false
        );
    double time_end = clock();
    printf("Compile to sass:\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    cuCheck( cuModuleLoadDataEx(&cu_module, sass, 0, NULL, NULL) );
    free(sass);

    cuCheck( cuModuleGetFunction(&cu_function, cu_module, "gemm_nt") );

    static const unsigned int block_dim = 256;

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
            cu_function,
            1, 1, 1,
            block_dim, 1, 1,
            0, 0, 
            args,
            NULL
        )
    );

    cuCheck( cuCtxSynchronize() );
    cuCheck( cuModuleUnload(cu_module) );
}

int
main() {
    double time_start, time_end;
    // The cmake plumbing already used the ptxinject cli tool compiled inside the 
    // project to process kernel.cu. The cuda was then compiled by nvcc as part of
    // the cmake process as well. INCBIN added the ptx to this file as g_annotated_ptx_data.

    time_start = clock();
    PtxInjectHandle ptx_inject;
    ptxInjectCheck(
        ptx_inject_create(
            &ptx_inject, 
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            g_annotated_ptx_data
        )
    );
    time_end = clock();

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

    printf("Parse/Process Inject PTX: %6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));
    printf("\n");
    // Print the stats of the injects that we're found
    print_ptx_inject_info(ptx_inject, ptx_inject_data_type_infos);
    printf("\n");

    size_t num_injects_found;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects_found) );

    // We should have 3 injects inside the gemm, multiply, accumulate and epilogue
    ASSERT(num_injects_found == 3);

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

    StackPtxRegister registers[] = {
        [REGISTER_MULTIPLY_OUTPUT_DIFF] =       {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_MULTIPLY_INPUT_V_A] =         {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_MULTIPLY_INPUT_V_B] =         {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_ACCUMULATE_OUTPUT_V_C_OUT] =  {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_ACCUMULATE_INPUT_V_C] =       {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_ACCUMULATE_INPUT_DIFF] =      {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_EPILOGUE_MODIFY_V_C] =        {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    };

    // We now need to ask which index in the stub buffer each inject needs to be at
    size_t multiply_func_idx;
    size_t accumulate_func_idx;
    size_t epilogue_func_idx;

    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "multiply", &multiply_func_idx, NULL, NULL) );
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "accumulate", &accumulate_func_idx, NULL, NULL) );
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "epilogue", &epilogue_func_idx, NULL, NULL) );

    char* multiply_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);
    char* accumulate_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);
    char* epilogue_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);

    const char* ptx_stubs[3];

    // Set up the buffers to go in to the proper indicies as specified by the inject_info function calls.
    ptx_stubs[multiply_func_idx] = multiply_stub_buffer;
    ptx_stubs[accumulate_func_idx] = accumulate_stub_buffer;
    ptx_stubs[epilogue_func_idx] = epilogue_stub_buffer;

    // Grab the "normalized" register names of the different operators inside the gemm.
    // The multiply, accumulate and epilogue operations.
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "diff", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_OUTPUT_DIFF].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "v_a", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_INPUT_V_A].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "v_b", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_INPUT_V_B].name) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "v_c_out", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_OUTPUT_V_C_OUT].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "v_c", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_INPUT_V_C].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "diff", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_INPUT_DIFF].name) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, epilogue_func_idx, "v_c", NULL, NULL, NULL, &registers[REGISTER_EPILOGUE_MODIFY_V_C].name) );

    // We need the multiply instructions to be assigned to the "diff" register name
    const size_t multiply_requests[] = { REGISTER_MULTIPLY_OUTPUT_DIFF };
    static const size_t num_multiply_requests = STACK_PTX_ARRAY_NUM_ELEMS(multiply_requests);

    // We need the accumulate instructions to be assigned to the "v_c_out" register name
    const size_t accumulate_requests[] = { REGISTER_ACCUMULATE_OUTPUT_V_C_OUT };
    static const size_t num_accumulate_requests = STACK_PTX_ARRAY_NUM_ELEMS(accumulate_requests);

    // We need the epilogue instructions to be assigned to the "v_c" register name.
    // This register has a modify type so it is used now as an output and later as an input as well.
    const size_t epilogue_requests[] = { REGISTER_ACCUMULATE_OUTPUT_V_C_OUT };
    static const size_t num_epilogue_requests = STACK_PTX_ARRAY_NUM_ELEMS(epilogue_requests);

    cuCheck( cuInit(0) );
    CUdevice cu_device;
    
    cuCheck( cuDeviceGet(&cu_device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    CUcontext cu_context;
    
    cuCheck( cuContextCreate(&cu_context, cu_device) );

    char* rendered_ptx;
    size_t num_bytes_written;

    static const int M = 128;
    static const int N = 128;
    static const int K = 8;
    static const int lda = 128;
    static const int ldb = 128;
    static const int ldc = 128;

    static const int matrix_print_limit = 5;

    float* h_a = (float*)malloc((size_t)M * (size_t)K * sizeof(float));
    float* h_b = (float*)malloc((size_t)N * (size_t)K * sizeof(float));
    float* h_c = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    float* h_c_gold = (float*)malloc((size_t)M * (size_t)N * sizeof(float));

    for (int j = 0; j < M*K; ++j) h_a[j] = 2*(rand() / (double)(RAND_MAX)) - 1;
    for (int j = 0; j < N*K; ++j) h_b[j] = 2*(rand() / (double)(RAND_MAX)) - 1;

    CUdeviceptr d_a;
    CUdeviceptr d_b;
    CUdeviceptr d_c;

    cuCheck( cuMemAlloc(&d_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_b, (size_t)N * (size_t)K * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_c, (size_t)M * (size_t)N * sizeof(float)) );

    cuCheck( cuMemcpyHtoD(d_a, h_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemcpyHtoD(d_b, h_b, (size_t)N * (size_t)K * sizeof(float)) );

    // Setup accumulate and epilogue once, they will stay the same throughout this example
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            (StackPtxInstruction[]) {
                stack_ptx_encode_input(REGISTER_ACCUMULATE_INPUT_V_C),
                stack_ptx_encode_input(REGISTER_ACCUMULATE_INPUT_DIFF),
                stack_ptx_encode_ptx_instruction_add_ftz_f32,
                stack_ptx_encode_return
            },
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            accumulate_requests,
            num_accumulate_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            accumulate_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            (StackPtxInstruction[]) {
                stack_ptx_encode_input(REGISTER_EPILOGUE_MODIFY_V_C),
                stack_ptx_encode_return
            },
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            epilogue_requests,
            num_epilogue_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            epilogue_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    // Multiply as gemm or `dot product`
    // diff = v_a * v_b
    static const StackPtxInstruction multiply_gemm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MULTIPLY_INPUT_V_A),
        stack_ptx_encode_input(REGISTER_MULTIPLY_INPUT_V_B),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_return
    };

    // Multiply as L1 Norm
    // diff = abs(v_a - v_b)
    static const StackPtxInstruction multiply_l1_norm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MULTIPLY_INPUT_V_A),
        stack_ptx_encode_input(REGISTER_MULTIPLY_INPUT_V_B),
        stack_ptx_encode_ptx_instruction_sub_ftz_f32,
        stack_ptx_encode_ptx_instruction_abs_ftz_f32,
        stack_ptx_encode_return
    };

    // Multiply as L2 Norm
    // diff = (v_a - v_b) * (v_a - v_b)
    static const StackPtxInstruction multiply_l2_norm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MULTIPLY_INPUT_V_A),
        stack_ptx_encode_input(REGISTER_MULTIPLY_INPUT_V_B),
        stack_ptx_encode_ptx_instruction_sub_ftz_f32,
        stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32), // Duplicate the ast value at the top of the f32 stack
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_return
    };
    
    // Set up the multiply operator to be just v_a * v_b for a mma gemm
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            multiply_gemm_instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            multiply_requests,
            num_multiply_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            multiply_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 3, &num_bytes_written);
    rendered_ptx[num_bytes_written-1] = '\0';
    time_end = clock();
    printf("MMA\n");
    printf("---------------------------------------------\n");
    printf("Render PTX:\t\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));
    
    // Run the kernel
    time_start = clock();
    run_custom_mma(
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc, 
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    time_end = clock();
    printf("Compile, Load, Run:\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    free(rendered_ptx);
    
    cuCheck( cuMemcpyDtoH(h_c, d_c, (size_t)M * (size_t)N * sizeof(float)) );

    // Compute a cpu version of the gemm
    gemm_gold(
        M, N, K,
        h_a, lda,
        h_b, ldb,
        h_c_gold, ldc
    );

    printf("\n");
    printf("MMA (Inject Kernel)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");
    printf("MMA (CPU Gold)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c_gold, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");

    // We can now easily run this also as an L1 norm kernel.
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            multiply_l1_norm_instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            multiply_requests,
            num_multiply_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            multiply_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 3, &num_bytes_written);
    rendered_ptx[num_bytes_written-1] = '\0';
    time_end = clock();
    printf("L1\n");
    printf("---------------------------------------------\n");
    printf("Render PTX:\t\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    // Run the kernel
    time_start = clock();
    run_custom_mma(
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc, 
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    time_end = clock();
    printf("Compile, Load, Run:\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    free(rendered_ptx);

    cuCheck( cuMemcpyDtoH(h_c, d_c, (size_t)M * (size_t)N * sizeof(float)) );

    // Compute a cpu version of the L1 norm
    l1_gold(
        M, N, K,
        h_a, lda,
        h_b, ldb,
        h_c_gold, ldc
    );

    printf("\n");
    printf("L1 (Inject Kernel)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");
    printf("L1 (CPU Gold)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c_gold, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");

    // Lastly we will run this as an L2 norm kernel
    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            multiply_l2_norm_instructions,
            registers, REGISTER_NUM_ENUMS,
            NULL, 0,
            multiply_requests,
            num_multiply_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            multiply_stub_buffer,
            STUB_BUFFER_SIZE,
            &num_bytes_written
        )
    );

    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 3, &num_bytes_written);
    rendered_ptx[num_bytes_written-1] = '\0';
    time_end = clock();
    printf("L2\n");
    printf("---------------------------------------------\n");
    printf("Render PTX:\t\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));
    
    time_start = clock();
    run_custom_mma(
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc, 
        device_compute_capability_major,
        device_compute_capability_minor,
        rendered_ptx,
        num_bytes_written
    );
    time_end = clock();
    printf("Compile, Load, Run:\t%6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));

    free(rendered_ptx);

    cuCheck( cuMemcpyDtoH(h_c, d_c, (size_t)M * (size_t)N * sizeof(float)) );

    l2_gold(
        M, N, K,
        h_a, lda,
        h_b, ldb,
        h_c_gold, ldc
    );

    printf("\n");
    printf("L2 (Inject Kernel)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");
    printf("L2 (CPU Gold)\n");
    printf("---------------------------------------------\n");
    print_matrix(M, N, h_c_gold, ldc, matrix_print_limit);
    printf("---------------------------------------------\n");

    cuCheck( cuMemFree(d_c) );
    cuCheck( cuMemFree(d_b) );
    cuCheck( cuMemFree(d_a) );

    free(h_c_gold);
    free(h_c);
    free(h_b);
    free(h_a);

    cuCheck( cuCtxDestroy(cu_context) );

    free(epilogue_stub_buffer);
    free(accumulate_stub_buffer);
    free(multiply_stub_buffer);

    free(stack_ptx_workspace);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );
    
    return 0;
}
