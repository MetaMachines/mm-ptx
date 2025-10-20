/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_DEBUG
#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>
#include <ptx_inject_default_generated_types.h>

#define STACK_PTX_DEBUG
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

#include <stack_ptx_default_generated_types.h>

#include <stack_ptx_default_info.h>

#include <cuda.h>
#include <time.h>

#define MAX_NUM_TIMING_STAGES 10
#define STUB_BUFFER_SIZE 1000000ull

#include <helpers.h>

static const char static_annotated_cuda[] = "   \n\
extern \"C\"                                    \n\
__global__                                      \n\
void                                            \n\
kernel(float *out) {                            \n\
	float x = 5.0f;                             \n\
	float y = 1.0f;                             \n\
    float z;                                    \n\
    for (int i = 0; i < 2; i++) {               \n\
    /* PTX_INJECT func                          \n\
        in  f32 x                               \n\
        mod f32 y                               \n\
        out f32 z                               \n\
    */                                          \n\
    }                                           \n\
    *out = z;                                   \n\
}                                               \n";

static const int execution_limit = 100;

int
main() {
    typedef struct {
        const char *annotated_cuda;
        char *rendered_cuda;
        const char *annotated_ptx;
        char *rendered_ptx;
        const char **ptx_stubs;
        size_t num_ptx_stubs;
        char *binary_image;
    } Buffers;

    int lines[MAX_NUM_TIMING_STAGES];
    double elapsed[MAX_NUM_TIMING_STAGES] = {0.0};

    Buffers buffers = {0};
    buffers.annotated_cuda = static_annotated_cuda;

    size_t stage = 0;

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

    size_t num_inject_sites;
    buffers.rendered_cuda = 
        process_cuda(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            buffers.annotated_cuda, 
            &num_inject_sites
        );

    ASSERT( num_inject_sites == 1 );

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

    CUdevice device;
    cuCheck( cuInit(0) );
    cuCheck( cuDeviceGet(&device, 0) );

    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(device, &device_compute_capability_major, &device_compute_capability_minor);

    buffers.annotated_ptx = nvrtc_compile(device_compute_capability_major, device_compute_capability_minor, buffers.rendered_cuda);

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

    PtxInjectHandle ptx_inject = {0};
    ptxInjectCheck(
        ptx_inject_create(
            &ptx_inject,
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            buffers.annotated_ptx
        )
    );

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    PtxInjectMutType mut_type_x;
    size_t data_type_idx_x;
    const char *register_name_x;

    PtxInjectMutType mut_type_y;
    size_t data_type_idx_y;
    const char *register_name_y;

    PtxInjectMutType mut_type_z;
    size_t data_type_idx_z;
    const char *register_name_z;

    ptxInjectCheck(
        ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_func_idx, "x",
            NULL,
            &mut_type_x, 
            &data_type_idx_x, 
            &register_name_x
        )
    );

    PtxInjectDataType data_type_x = (PtxInjectDataType)data_type_idx_x;

    ASSERT( mut_type_x == PTX_INJECT_MUT_TYPE_IN );
    ASSERT( data_type_x == PTX_INJECT_DATA_TYPE_F32 );

    ptxInjectCheck(
        ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_func_idx, "y",
            NULL,
            &mut_type_y, 
            &data_type_idx_y, 
            &register_name_y
        )
    );

    PtxInjectDataType data_type_y = (PtxInjectDataType)data_type_idx_y;

    ASSERT( mut_type_y == PTX_INJECT_MUT_TYPE_MOD );
    ASSERT( data_type_y == PTX_INJECT_DATA_TYPE_F32 );

    ptxInjectCheck(
        ptx_inject_variable_info_by_name(
            ptx_inject,
            inject_func_idx, "z",
            NULL, 
            &mut_type_z, 
            &data_type_idx_z, 
            &register_name_z
        )
    );

    PtxInjectDataType data_type_z = (PtxInjectDataType)data_type_idx_z;

    ASSERT( mut_type_z == PTX_INJECT_MUT_TYPE_OUT );
    ASSERT( data_type_z == PTX_INJECT_DATA_TYPE_F32 );

    static const size_t num_ptx_stubs = 1;

    buffers.ptx_stubs = (const char **)malloc(num_ptx_stubs * sizeof(char*));
    buffers.num_ptx_stubs = num_ptx_stubs;

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

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

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

    enum Register{
        REGISTER_IN_X,
        REGISTER_MOD_Y,
        REGISTER_OUT_Z,
        REGISTER_NUM_ENUMS
    };

    const StackPtxRegister registers[] = {
        [REGISTER_IN_X] = {.name = register_name_x, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_MOD_Y] = {.name = register_name_y, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_OUT_Z] = {.name = register_name_z, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    };
    static const size_t num_registers = REGISTER_NUM_ENUMS;

    static const StackPtxInstruction instructions[] = {
        stack_ptx_encode_input(REGISTER_IN_X),
        stack_ptx_encode_input(REGISTER_MOD_Y),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_input(REGISTER_IN_X),
        stack_ptx_encode_input(REGISTER_MOD_Y),
        stack_ptx_encode_ptx_instruction_add_ftz_f32,
        stack_ptx_encode_return
    };

    const size_t requests[] = { REGISTER_MOD_Y, REGISTER_OUT_Z };
    static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

    size_t stack_ptx_buffer_size;
    stackPtxCheck( 
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers, num_registers,
            NULL, 0,
            requests,
            num_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            NULL,
            0ull,
            &stack_ptx_buffer_size
        )
    );

    stack_ptx_buffer_size++;
    char *stack_ptx_buffer = (char*)malloc(stack_ptx_buffer_size);

    stackPtxCheck(
        stack_ptx_compile(
            &compiler_info,
            &stack_ptx_stack_info,
            instructions,
            registers, num_registers,
            NULL, 0,
            requests,
            num_requests,
            execution_limit,
            stack_ptx_workspace,
            stack_ptx_workspace_size,
            stack_ptx_buffer,
            stack_ptx_buffer_size,
            &stack_ptx_buffer_size
        )
    );

    free(stack_ptx_workspace);

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

    buffers.ptx_stubs[0] = stack_ptx_buffer;

    size_t rendered_ptx_bytes_written;
    buffers.rendered_ptx = 
        render_injected_ptx(
            ptx_inject, 
            buffers.ptx_stubs, 
            num_ptx_stubs, 
            &rendered_ptx_bytes_written
        );

    ptxInjectCheck(
        ptx_inject_destroy(ptx_inject)
    );

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

    printf("%s\n", buffers.rendered_ptx);

    buffers.binary_image = 
        nvptx_compile(
            device_compute_capability_major, device_compute_capability_minor, 
            buffers.rendered_ptx, 
            rendered_ptx_bytes_written, 
            false
        );

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    cuCheck( cuContextCreate(&context, device) );
    cuCheck( cuModuleLoadDataEx(&module, buffers.binary_image, 0, 0, 0) );
    cuCheck( cuModuleGetFunction(&kernel, module, "kernel") );

    CUdeviceptr d_out;
    cuCheck( cuMemAlloc(&d_out, sizeof(float)) );

    void* args[] = {
        (void*)&d_out
    };

    cuCheck( 
        cuLaunchKernel(kernel,
            1, 1, 1,
            1, 1, 1,
            0, NULL,
            args, 0
        )
    );
    cuCheck( cuCtxSynchronize() );

    float h_out;
    cuCheck( cuMemcpyDtoH(&h_out, d_out, sizeof(float)) );

    printf("result (expected 11.0): %f\n\n", h_out);
    ASSERT( h_out == 11.0f );

    cuCheck( cuMemFree(d_out) );
    cuCheck( cuModuleUnload(module) );
    cuCheck( cuCtxDestroy(context) );

    elapsed[stage++] = clock(); lines[stage-1] = __LINE__;

    for (size_t i = 1; i < stage; i++) {
        double diff = elapsed[i] - elapsed[i-1];
        diff = diff / CLOCKS_PER_SEC * 1000000.0;
        printf("Stage %d: %6d micros, %d\n", (int)(i - 1) , (int)diff, lines[i]);
    }

    return 0;
}
