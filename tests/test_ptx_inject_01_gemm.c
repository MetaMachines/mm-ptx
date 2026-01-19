/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_helper.h>
#include <nvptx_helper.h>
#include <cuda_helper.h>
#include <check_result_helper.h>
#include <mma_helper.h>

#include <cuda.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* Use incbin to bring the code from kernel.ptx, allows easy editing of cuda source
*   is replaced with g_annotated_ptx_data
*/
INCTXT(annotated_ptx, PTX_KERNEL);

#define STUB_BUFFER_SIZE 1000000ull

static const float matrix_tol = 1e-3f;

static
bool
has_cuda_device(void) {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return false;
    }
    return count > 0;
}

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

    void* sass =
        nvptx_compile(
            device_compute_capability_major,
            device_compute_capability_minor,
            rendered_ptx,
            rendered_ptx_num_bytes,
            NULL,
            false
        );

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

typedef void (*GoldFn)(
    int M, int N, int K,
    float* h_a, int lda,
    float* h_b, int ldb,
    float* h_c, int ldc
);

static
void
run_and_validate(
    const char* label,
    PtxInjectHandle ptx_inject,
    const char* const* ptx_stubs,
    size_t num_ptx_stubs,
    int M, int N, int K,
    CUdeviceptr d_a, int lda,
    CUdeviceptr d_b, int ldb,
    CUdeviceptr d_c, int ldc,
    int device_compute_capability_major,
    int device_compute_capability_minor,
    float* h_a,
    float* h_b,
    float* h_c,
    float* h_c_gold,
    GoldFn gold_fn
) {
    size_t num_bytes_written = 0;
    char* rendered_ptx =
        render_injected_ptx(ptx_inject, ptx_stubs, num_ptx_stubs, &num_bytes_written);

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

    free(rendered_ptx);

    cuCheck( cuMemcpyDtoH(h_c, d_c, (size_t)M * (size_t)N * sizeof(float)) );

    gold_fn(
        M, N, K,
        h_a, lda,
        h_b, ldb,
        h_c_gold, ldc
    );

    float max_diff = matrix_max_abs_diff(M, N, h_c, ldc, h_c_gold, ldc);
    if (max_diff > matrix_tol) {
        fprintf(stderr, "%s max diff %g (tol %g)\n", label, max_diff, matrix_tol);
    }
    ASSERT(max_diff <= matrix_tol);
}

int
main() {
    if (!has_cuda_device()) {
        fprintf(stderr, "SKIP: no CUDA device available\n");
        return 77;
    }

    srand(0);

    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    size_t num_injects_found;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects_found) );
    ASSERT(num_injects_found == 2);

    size_t mma_func_idx;
    size_t epilogue_func_idx;

    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "mma", &mma_func_idx, NULL, NULL) );
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "epilogue", &epilogue_func_idx, NULL, NULL) );
    ASSERT(mma_func_idx < num_injects_found);
    ASSERT(epilogue_func_idx < num_injects_found);

    const char* mma_register_name_v_a;
    const char* mma_register_name_v_b;
    const char* mma_register_name_v_c;

    const char* epilogue_register_name_v_c_in;
    const char* epilogue_register_name_v_c_out;

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, mma_func_idx, "v_a", NULL, &mma_register_name_v_a, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, mma_func_idx, "v_b", NULL, &mma_register_name_v_b, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, mma_func_idx, "v_c", NULL, &mma_register_name_v_c, NULL, NULL, NULL) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, epilogue_func_idx, "v_c_in", NULL, &epilogue_register_name_v_c_in, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, epilogue_func_idx, "v_c_out", NULL, &epilogue_register_name_v_c_out, NULL, NULL, NULL) );

    char* mma_stub_buffer = (char *)malloc(STUB_BUFFER_SIZE);
    char* epilogue_stub_buffer = (char *)malloc(STUB_BUFFER_SIZE);
    ASSERT(mma_stub_buffer != NULL);
    ASSERT(epilogue_stub_buffer != NULL);

    const char* ptx_stubs[2] = {0};
    ptx_stubs[mma_func_idx] = mma_stub_buffer;
    ptx_stubs[epilogue_func_idx] = epilogue_stub_buffer;

    cuCheck( cuInit(0) );
    CUdevice device;
    cuCheck( cuDeviceGet(&device, 0) );

    int device_compute_capability_major;
    int device_compute_capability_minor;
    get_device_capability(device, &device_compute_capability_major, &device_compute_capability_minor);

    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, device) );

    static const int M = 128;
    static const int N = 128;
    static const int K = 8;
    static const int lda = 128;
    static const int ldb = 128;
    static const int ldc = 128;

    float* h_a = (float*)malloc((size_t)M * (size_t)K * sizeof(float));
    float* h_b = (float*)malloc((size_t)N * (size_t)K * sizeof(float));
    float* h_c = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    float* h_c_gold = (float*)malloc((size_t)M * (size_t)N * sizeof(float));
    ASSERT(h_a != NULL);
    ASSERT(h_b != NULL);
    ASSERT(h_c != NULL);
    ASSERT(h_c_gold != NULL);

    for (int j = 0; j < M*K; ++j) {
        h_a[j] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
    for (int j = 0; j < N*K; ++j) {
        h_b[j] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }

    CUdeviceptr d_a;
    CUdeviceptr d_b;
    CUdeviceptr d_c;

    cuCheck( cuMemAlloc(&d_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_b, (size_t)N * (size_t)K * sizeof(float)) );
    cuCheck( cuMemAlloc(&d_c, (size_t)M * (size_t)N * sizeof(float)) );

    cuCheck( cuMemcpyHtoD(d_a, h_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemcpyHtoD(d_b, h_b, (size_t)N * (size_t)K * sizeof(float)) );

    snprintf(mma_stub_buffer, STUB_BUFFER_SIZE,
        "\tfma.rn.ftz.f32 %%%3$s, %%%2$s, %%%1$s, %%%3$s;",
        mma_register_name_v_a,
        mma_register_name_v_b,
        mma_register_name_v_c
    );

    snprintf(epilogue_stub_buffer, STUB_BUFFER_SIZE,
        "\tmov.f32 %%%2$s, %%%1$s;",
        epilogue_register_name_v_c_in,
        epilogue_register_name_v_c_out
    );

    run_and_validate(
        "MMA",
        ptx_inject,
        ptx_stubs,
        num_injects_found,
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc,
        device_compute_capability_major,
        device_compute_capability_minor,
        h_a,
        h_b,
        h_c,
        h_c_gold,
        gemm_gold
    );

    snprintf(mma_stub_buffer, STUB_BUFFER_SIZE,
        "\tsub.ftz.f32 %%%1$s, %%%2$s, %%%1$s;\n"
        "\tabs.ftz.f32 %%%1$s, %%%1$s;\n"
        "\tadd.ftz.f32 %%%3$s, %%%3$s, %%%1$s;",
        mma_register_name_v_a,
        mma_register_name_v_b,
        mma_register_name_v_c
    );

    run_and_validate(
        "L1",
        ptx_inject,
        ptx_stubs,
        num_injects_found,
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc,
        device_compute_capability_major,
        device_compute_capability_minor,
        h_a,
        h_b,
        h_c,
        h_c_gold,
        l1_gold
    );

    snprintf(mma_stub_buffer, STUB_BUFFER_SIZE,
        "\tsub.ftz.f32 %%%1$s, %%%2$s, %%%1$s;\n"
        "\tmul.ftz.f32 %%%1$s, %%%1$s, %%%1$s;\n"
        "\tadd.ftz.f32 %%%3$s, %%%3$s, %%%1$s;",
        mma_register_name_v_a,
        mma_register_name_v_b,
        mma_register_name_v_c
    );

    run_and_validate(
        "L2",
        ptx_inject,
        ptx_stubs,
        num_injects_found,
        M, N, K,
        d_a, lda,
        d_b, ldb,
        d_c, ldc,
        device_compute_capability_major,
        device_compute_capability_minor,
        h_a,
        h_b,
        h_c,
        h_c_gold,
        l2_gold
    );

    cuCheck( cuMemFree(d_c) );
    cuCheck( cuMemFree(d_b) );
    cuCheck( cuMemFree(d_a) );

    free(h_c_gold);
    free(h_c);
    free(h_b);
    free(h_a);

    cuCheck( cuCtxDestroy(cu_context) );

    free(epilogue_stub_buffer);
    free(mma_stub_buffer);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    return 0;
}
