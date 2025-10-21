/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_default_generated_types.h>

#include <cuda.h>
#include <helpers.h>
#include <math.h>
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
INCTXT(annotated_ptx, XSTRING(PTX_KERNEL));

#define STUB_BUFFER_SIZE 1000000ull

#define CLOCK_MULTIPLIER 1000000.0

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

    printf("Parse/Process Inject PTX: %6d micros\n", (int)((time_end - time_start) / CLOCKS_PER_SEC * CLOCK_MULTIPLIER));
    printf("\n");
    // Print the stats of the injects that we're found
    print_ptx_inject_info(ptx_inject, ptx_inject_data_type_infos);
    printf("\n");

    size_t num_injects_found;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects_found) );

    // We should have 3 injects inside the gemm, multiply, accumulate and epilogue
    ASSERT(num_injects_found == 3);

    const char* multiply_register_name_diff;
    const char* multiply_register_name_v_a;
    const char* multiply_register_name_v_b;

    const char* accumulate_register_name_v_c_out;
    const char* accumulate_register_name_v_c;
    const char* accumulate_register_name_diff;

    const char* epilogue_register_name_v_c;

    // We now need to ask which index in the stub buffer each inject needs to be at
    size_t multiply_func_idx;
    size_t accumulate_func_idx;
    size_t epilogue_func_idx;

    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "multiply", &multiply_func_idx, NULL, NULL) );
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "accumulate", &accumulate_func_idx, NULL, NULL) );
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "epilogue", &epilogue_func_idx, NULL, NULL) );

    // Grab the "normalized" register names of the different operators inside the gemm.
    // The multiply, accumulate and epilogue operations.
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "diff", NULL, NULL, NULL, &multiply_register_name_diff) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "v_a", NULL, NULL, NULL, &multiply_register_name_v_a) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "v_b", NULL, NULL, NULL, &multiply_register_name_v_b) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "v_c_out", NULL, NULL, NULL, &accumulate_register_name_v_c_out) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "v_c", NULL, NULL, NULL, &accumulate_register_name_v_c) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "diff", NULL, NULL, NULL, &accumulate_register_name_diff) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, epilogue_func_idx, "v_c", NULL, NULL, NULL, &epilogue_register_name_v_c) );

    char *multiply_stub_buffer = (char *)malloc(STUB_BUFFER_SIZE);
    char *accumulate_stub_buffer = (char *)malloc(STUB_BUFFER_SIZE);
    char *epilogue_stub_buffer = (char *)malloc(STUB_BUFFER_SIZE);

    const char* ptx_stubs[3];

    ptx_stubs[multiply_func_idx] = multiply_stub_buffer;
    ptx_stubs[accumulate_func_idx] = accumulate_stub_buffer;
    ptx_stubs[epilogue_func_idx] = epilogue_stub_buffer;

    cuCheck( cuInit(0) );
    CUdevice device;
    
    cuCheck( cuDeviceGet(&device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    CUcontext cu_context;
    
    cuCheck( cuContextCreate(&cu_context, device) );

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

    // Now that everything is setup, lets run this kernel as a typical gemm mma

    // For multiply we just multiply
    snprintf(multiply_stub_buffer, STUB_BUFFER_SIZE, 
        "\tmul.ftz.f32 %%%3$s, %%%2$s, %%%1$s;",
        multiply_register_name_v_a,
        multiply_register_name_v_b,
        multiply_register_name_diff
    );

    // For accumulate we add
    snprintf(accumulate_stub_buffer, STUB_BUFFER_SIZE, 
        "\tadd.ftz.f32 %%%3$s, %%%2$s, %%%1$s;",
        accumulate_register_name_v_c,
        accumulate_register_name_diff,
        accumulate_register_name_v_c_out
    );

    // For epilogue we do nothing
    snprintf(epilogue_stub_buffer, STUB_BUFFER_SIZE, 
        "\t"
        // accumulate_register_name_v_c_out,
        // accumulate_register_name_diff,
        // accumulate_register_name_v_c
    );

    // Take the stubs and use it to render new ptx
    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 3, &num_bytes_written);
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

    // We can now easily run this as an L1 norm kernel.

    // The multiply is now abs(x-y)
    snprintf(multiply_stub_buffer, STUB_BUFFER_SIZE, 
        "\tsub.ftz.f32 %%%3$s, %%%2$s, %%%1$s;\n"
        "\tabs.ftz.f32 %%%3$s, %%%3$s;",
        multiply_register_name_v_a,
        multiply_register_name_v_b,
        multiply_register_name_diff
    );

    // The accumulate is still just add
    snprintf(accumulate_stub_buffer, STUB_BUFFER_SIZE, 
        "\tadd.ftz.f32 %%%3$s, %%%2$s, %%%1$s;",
        accumulate_register_name_v_c,
        accumulate_register_name_diff,
        accumulate_register_name_v_c_out
    );

    // For epilogue we do nothing
    snprintf(epilogue_stub_buffer, STUB_BUFFER_SIZE, 
        "\t"
        // accumulate_register_name_v_c_out,
        // accumulate_register_name_diff,
        // accumulate_register_name_v_c
    );

    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 3, &num_bytes_written);
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

    // For multiply we do (x-y)^2
    snprintf(multiply_stub_buffer, STUB_BUFFER_SIZE, 
        "\tsub.ftz.f32 %%%3$s, %%%2$s, %%%1$s;\n"
        "\tmul.ftz.f32 %%%3$s, %%%3$s, %%%3$s;",
        multiply_register_name_v_a,
        multiply_register_name_v_b,
        multiply_register_name_diff
    );

    // For accumulate we still add
    snprintf(accumulate_stub_buffer, STUB_BUFFER_SIZE, 
        "\tadd.ftz.f32 %%%3$s, %%%2$s, %%%1$s;",
        accumulate_register_name_v_c,
        accumulate_register_name_diff,
        accumulate_register_name_v_c_out
    );

    // For epilogue we do nothing
    snprintf(epilogue_stub_buffer, STUB_BUFFER_SIZE, 
        "\t"
        // accumulate_register_name_v_c_out,
        // accumulate_register_name_diff,
        // accumulate_register_name_v_c
    );

    time_start = clock();
    rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 3, &num_bytes_written);
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

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );
    
    return 0;
}
