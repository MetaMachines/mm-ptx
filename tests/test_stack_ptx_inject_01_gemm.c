/*
 * SPDX-FileCopyrightText: 2026 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_example_descriptions.h>
#include <stack_ptx_default_info.h>

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

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* Use incbin to bring the code from kernel.ptx, allows easy editing of cuda source
*   is replaced with g_annotated_ptx_data
*/
INCTXT(annotated_ptx, PTX_KERNEL);

#define STUB_BUFFER_SIZE 1000000ull

static const float matrix_tol = 1e-3f;
static const int execution_limit = 100;

enum InjectSite {
    INJECT_MMA,
    INJECT_EPILOGUE,
    INJECT_NUM_SITES
};

typedef struct {
    const char* name;
    size_t idx;
} InjectSiteInfo;

static InjectSiteInfo g_inject_sites[INJECT_NUM_SITES] = {
    [INJECT_MMA] = { "mma", 0 },
    [INJECT_EPILOGUE] = { "epilogue", 0 },
};

enum Register {
    REGISTER_MMA_V_A_IN,
    REGISTER_MMA_V_B_IN,
    REGISTER_MMA_V_C_MOD,
    REGISTER_EPILOGUE_V_C_IN,
    REGISTER_EPILOGUE_V_C_OUT,
    REGISTER_NUM_ENUMS
};

static StackPtxRegister registers[] = {
    [REGISTER_MMA_V_A_IN] =         { NULL, STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_MMA_V_B_IN] =         { NULL, STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_MMA_V_C_MOD] =        { NULL, STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_EPILOGUE_V_C_IN] =    { NULL, STACK_PTX_STACK_TYPE_F32 },
    [REGISTER_EPILOGUE_V_C_OUT] =   { NULL, STACK_PTX_STACK_TYPE_F32 },
};
static const size_t num_registers = REGISTER_NUM_ENUMS;

typedef struct {
    enum InjectSite site;
    const char* var_name;
    enum Register reg;
} RegisterBinding;

static const RegisterBinding g_register_bindings[] = {
    { INJECT_MMA,      "v_a",     REGISTER_MMA_V_A_IN },
    { INJECT_MMA,      "v_b",     REGISTER_MMA_V_B_IN },
    { INJECT_MMA,      "v_c",     REGISTER_MMA_V_C_MOD },
    { INJECT_EPILOGUE, "v_c_in",  REGISTER_EPILOGUE_V_C_IN },
    { INJECT_EPILOGUE, "v_c_out", REGISTER_EPILOGUE_V_C_OUT },
};

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

static
void
compile_stack_ptx_stub(
    const StackPtxInstruction* instructions,
    const size_t* requests,
    size_t num_requests,
    void* stack_ptx_workspace,
    size_t stack_ptx_workspace_size,
    char* stub_buffer,
    size_t stub_buffer_size
) {
    size_t num_bytes_written = 0;
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
            stub_buffer,
            stub_buffer_size,
            &num_bytes_written
        )
    );
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

    size_t stack_ptx_workspace_size;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &stack_ptx_workspace_size
        )
    );

    void* stack_ptx_workspace = malloc(stack_ptx_workspace_size);
    ASSERT(stack_ptx_workspace != NULL);

    size_t num_injects_found;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects_found) );
    ASSERT(num_injects_found == INJECT_NUM_SITES);

    for (size_t s = 0; s < INJECT_NUM_SITES; ++s) {
        ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, g_inject_sites[s].name, &g_inject_sites[s].idx, NULL, NULL) );
    }

    for (size_t i = 0; i < STACK_PTX_ARRAY_NUM_ELEMS(g_register_bindings); ++i) {
        const RegisterBinding* binding = &g_register_bindings[i];
        size_t inject_idx = g_inject_sites[binding->site].idx;
        ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_idx, binding->var_name, NULL, &registers[binding->reg].name, NULL, NULL, NULL) );
    }

    char* mma_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);
    char* epilogue_stub_buffer = (char*)malloc(STUB_BUFFER_SIZE);
    ASSERT(mma_stub_buffer != NULL);
    ASSERT(epilogue_stub_buffer != NULL);

    const char* ptx_stubs[INJECT_NUM_SITES] = {0};
    ptx_stubs[g_inject_sites[INJECT_MMA].idx] = mma_stub_buffer;
    ptx_stubs[g_inject_sites[INJECT_EPILOGUE].idx] = epilogue_stub_buffer;

    static const size_t mma_requests[] = { REGISTER_MMA_V_C_MOD };
    static const size_t num_mma_requests = STACK_PTX_ARRAY_NUM_ELEMS(mma_requests);

    static const size_t epilogue_requests[] = { REGISTER_EPILOGUE_V_C_OUT };
    static const size_t num_epilogue_requests = STACK_PTX_ARRAY_NUM_ELEMS(epilogue_requests);

    static const StackPtxInstruction epilogue_instructions[] = {
        stack_ptx_encode_input(REGISTER_EPILOGUE_V_C_IN),
        stack_ptx_encode_return
    };

    compile_stack_ptx_stub(
        epilogue_instructions,
        epilogue_requests,
        num_epilogue_requests,
        stack_ptx_workspace,
        stack_ptx_workspace_size,
        epilogue_stub_buffer,
        STUB_BUFFER_SIZE
    );

    static const StackPtxInstruction mma_gemm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MMA_V_C_MOD),
        stack_ptx_encode_input(REGISTER_MMA_V_B_IN),
        stack_ptx_encode_input(REGISTER_MMA_V_A_IN),
        stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction mma_l1_norm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MMA_V_A_IN),
        stack_ptx_encode_input(REGISTER_MMA_V_B_IN),
        stack_ptx_encode_ptx_instruction_sub_ftz_f32,
        stack_ptx_encode_ptx_instruction_abs_ftz_f32,
        stack_ptx_encode_input(REGISTER_MMA_V_C_MOD),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    static const StackPtxInstruction mma_l2_norm_instructions[] = {
        stack_ptx_encode_input(REGISTER_MMA_V_A_IN),
        stack_ptx_encode_input(REGISTER_MMA_V_B_IN),
        stack_ptx_encode_ptx_instruction_sub_ftz_f32,
        stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32),
        stack_ptx_encode_ptx_instruction_mul_ftz_f32,
        stack_ptx_encode_input(REGISTER_MMA_V_C_MOD),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

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

    compile_stack_ptx_stub(
        mma_gemm_instructions,
        mma_requests,
        num_mma_requests,
        stack_ptx_workspace,
        stack_ptx_workspace_size,
        mma_stub_buffer,
        STUB_BUFFER_SIZE
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

    compile_stack_ptx_stub(
        mma_l1_norm_instructions,
        mma_requests,
        num_mma_requests,
        stack_ptx_workspace,
        stack_ptx_workspace_size,
        mma_stub_buffer,
        STUB_BUFFER_SIZE
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

    compile_stack_ptx_stub(
        mma_l2_norm_instructions,
        mma_requests,
        num_mma_requests,
        stack_ptx_workspace,
        stack_ptx_workspace_size,
        mma_stub_buffer,
        STUB_BUFFER_SIZE
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

    free(stack_ptx_workspace);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    return 0;
}
