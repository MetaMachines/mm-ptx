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

#include <check_result_helper.h>
#include <ptx_inject_helper.h>
#include <nvptx_helper.h>
#include <cuda_helper.h>
#include <cuda.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

/* Use incbin to bring the code from kernel.ptx, allows easy editing of cuda source
*   is replaced with g_annotated_ptx_data
*/
INCTXT(annotated_ptx, PTX_KERNEL);

#define PHILOX_W32_0   (0x9E3779B9)
#define PHILOX_W32_1   (0xBB67AE85)
#define PHILOX_M4x32_0 (0xD2511F53)
#define PHILOX_M4x32_1 (0xCD9E8D57)

static const int execution_limit = 100000;

#define PHILOX_REGISTER_DECL    \
    REGISTER_R_KEY_X,           \
    REGISTER_R_KEY_Y,           \
    REGISTER_R_CTR_X,           \
    REGISTER_R_CTR_Y,           \
    REGISTER_R_CTR_Z,           \
    REGISTER_R_CTR_W

#define PHILOX_REGISTER_IMPL                                                        \
    [REGISTER_R_KEY_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX},  \
    [REGISTER_R_KEY_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX},  \
    [REGISTER_R_CTR_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX},  \
    [REGISTER_R_CTR_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX},  \
    [REGISTER_R_CTR_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX},  \
    [REGISTER_R_CTR_W] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_PHILOX}


typedef enum {
    ORIG_KEY_X_IDX = 10,
    ORIG_KEY_Y_IDX = 11,
    ORIG_CTR_X_IDX = 12,
    ORIG_CTR_Y_IDX = 13,
    ORIG_CTR_Z_IDX = 14,
    ORIG_CTR_W_IDX = 15,
    MUT_KEY_X_IDX = 16,
    MUT_KEY_Y_IDX = 17,
    MUT_CTR_X_IDX = 18,
    MUT_CTR_Y_IDX = 19,
    MUT_CTR_Z_IDX = 20,
    MUT_CTR_W_IDX = 21,
    SWAP_KEY_X_IDX = 22,
    SWAP_KEY_Y_IDX = 23,
    SWAP_CTR_X_IDX = 24,
    SWAP_CTR_Y_IDX = 25,
    SWAP_CTR_Z_IDX = 26,
    SWAP_CTR_W_IDX = 27,
} PhiloxIDX;

enum Register {
    PHILOX_REGISTER_DECL,
    REGISTER_X,
    REGISTER_Y,
    REGISTER_Z,
    REGISTER_W,
    REGISTER_NUM_ENUMS
};

static StackPtxRegister registers[] = {
    PHILOX_REGISTER_IMPL,
    [REGISTER_X] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
    [REGISTER_Y] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
    [REGISTER_Z] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
    [REGISTER_W] = {.name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_U32},
};
static const size_t num_registers = REGISTER_NUM_ENUMS;

enum RoutineIdx {
    ROUTINE_PHILOX_PUSH_STATE,
    ROUTINE_PHILOX_CURAND,
    ROUTINE_PHILOX_ROUND,
    NUM_ROUTINES
};

static const StackPtxInstruction* philox_routines[] = {
    [ROUTINE_PHILOX_PUSH_STATE] = (StackPtxInstruction[]){
        stack_ptx_encode_input(REGISTER_R_KEY_Y),
        stack_ptx_encode_input(REGISTER_R_KEY_X),
        stack_ptx_encode_input(REGISTER_R_CTR_W),
        stack_ptx_encode_input(REGISTER_R_CTR_Z),
        stack_ptx_encode_input(REGISTER_R_CTR_Y),
        stack_ptx_encode_input(REGISTER_R_CTR_X),
        stack_ptx_encode_return
    },
    [ROUTINE_PHILOX_ROUND] = (StackPtxInstruction[]) {
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, MUT_CTR_X_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, MUT_CTR_Y_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, MUT_CTR_Z_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, MUT_CTR_W_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, MUT_KEY_X_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, MUT_KEY_Y_IDX),

        // key_y
        stack_ptx_encode_load(MUT_KEY_Y_IDX),
        stack_ptx_encode_constant_philox(PHILOX_W32_1),
        stack_ptx_encode_ptx_instruction_add_philox,

        // key_x
        stack_ptx_encode_load(MUT_KEY_X_IDX),
        stack_ptx_encode_constant_philox(PHILOX_W32_0),
        stack_ptx_encode_ptx_instruction_add_philox,

        // ctr_w
        stack_ptx_encode_load(MUT_CTR_X_IDX),
        stack_ptx_encode_constant_philox(PHILOX_M4x32_0),
        stack_ptx_encode_ptx_instruction_mul_lo_philox,

        // ctr_z
        stack_ptx_encode_load(MUT_CTR_X_IDX),
        stack_ptx_encode_constant_philox(PHILOX_M4x32_0),
        stack_ptx_encode_ptx_instruction_mul_hi_philox,
        stack_ptx_encode_load(MUT_CTR_W_IDX),
        stack_ptx_encode_ptx_instruction_xor_philox,
        stack_ptx_encode_load(MUT_KEY_Y_IDX),
        stack_ptx_encode_ptx_instruction_xor_philox,

        // ctr_y
        stack_ptx_encode_load(MUT_CTR_Z_IDX),
        stack_ptx_encode_constant_philox(PHILOX_M4x32_1),
        stack_ptx_encode_ptx_instruction_mul_lo_philox,

        // ctr_x
        stack_ptx_encode_load(MUT_CTR_Z_IDX),
        stack_ptx_encode_constant_philox(PHILOX_M4x32_1),
        stack_ptx_encode_ptx_instruction_mul_hi_philox,
        stack_ptx_encode_load(MUT_CTR_Y_IDX),
        stack_ptx_encode_ptx_instruction_xor_philox,
        stack_ptx_encode_load(MUT_KEY_X_IDX),
        stack_ptx_encode_ptx_instruction_xor_philox,
        stack_ptx_encode_return
    },
    [ROUTINE_PHILOX_CURAND] = (StackPtxInstruction[]){
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, ORIG_CTR_X_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, ORIG_CTR_Y_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, ORIG_CTR_Z_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, ORIG_CTR_W_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, ORIG_KEY_X_IDX),
        stack_ptx_encode_store(STACK_PTX_STACK_TYPE_PHILOX, ORIG_KEY_Y_IDX),

        stack_ptx_encode_load(ORIG_KEY_Y_IDX),
        stack_ptx_encode_load(ORIG_KEY_X_IDX),
        stack_ptx_encode_load(ORIG_CTR_W_IDX),
        stack_ptx_encode_load(ORIG_CTR_Z_IDX),
        stack_ptx_encode_load(ORIG_CTR_Y_IDX),
        stack_ptx_encode_load(ORIG_CTR_X_IDX),

        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_ROUND),

        stack_ptx_encode_ptx_instruction_mov_philox,
        stack_ptx_encode_ptx_instruction_mov_philox,
        stack_ptx_encode_ptx_instruction_mov_philox,
        stack_ptx_encode_ptx_instruction_mov_philox,

        stack_ptx_encode_meta_constant(2),
        stack_ptx_encode_meta_drop(STACK_PTX_STACK_TYPE_PHILOX),

        stack_ptx_encode_load(ORIG_KEY_Y_IDX),
        stack_ptx_encode_load(ORIG_KEY_X_IDX),
        stack_ptx_encode_load(ORIG_CTR_W_IDX),
        stack_ptx_encode_load(ORIG_CTR_Z_IDX),
        stack_ptx_encode_load(ORIG_CTR_Y_IDX),
        stack_ptx_encode_load(ORIG_CTR_X_IDX),

        stack_ptx_encode_constant_philox(1),
        stack_ptx_encode_ptx_instruction_add_philox,
        stack_ptx_encode_return
    },
};

static
unsigned int
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
            philox_routines, 
            NUM_ROUTINES,
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
            philox_routines, 
            NUM_ROUTINES,
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

    // We should now see the add instruction inside the ptx.
    // printf(
    //     "---------------------------------------------\n"
    //     "%s"
    //     "---------------------------------------------\n",
    //     rendered_ptx
    // );
    
    // We can now compile this ptx to sass
    void* sass = nvptx_compile(device_compute_capability_major, device_compute_capability_minor, rendered_ptx, num_bytes_written, NULL, false);

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

    unsigned int h_out;
    cuCheck( cuMemcpyDtoH(&h_out, d_out, sizeof(unsigned int)) );

    return h_out;
}

static
PtxInjectResult
ptx_inject_philox_populate_registers(
    PtxInjectHandle ptx_inject,
    size_t inject_func_idx,
    StackPtxRegister* registers
) {
    PtxInjectResult result;
    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_key_x", NULL, &registers[REGISTER_R_KEY_X].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_key_y", NULL, &registers[REGISTER_R_KEY_Y].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_ctr_x", NULL, &registers[REGISTER_R_CTR_X].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_ctr_y", NULL, &registers[REGISTER_R_CTR_Y].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_ctr_z", NULL, &registers[REGISTER_R_CTR_Z].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    result = ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "philox_ctr_w", NULL, &registers[REGISTER_R_CTR_W].name, NULL, NULL, NULL);
    if (result != PTX_INJECT_SUCCESS) return result;

    return result;
}

int
main() {
    // printf("\nAnnotated PTX\n"
    //     "---------------------------------------------\n"
    //     "%.*s"
    //     "---------------------------------------------\n\n",
    //      g_annotated_ptx_size, g_annotated_ptx_data
    // );

    PtxInjectHandle ptx_inject;
    ptxInjectCheck( ptx_inject_create(&ptx_inject, g_annotated_ptx_data) );

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    ptxInjectCheck( ptx_inject_philox_populate_registers(ptx_inject, inject_func_idx, registers) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "x", NULL, &registers[REGISTER_X].name, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "y", NULL, &registers[REGISTER_Y].name, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "z", NULL, &registers[REGISTER_Z].name, NULL, NULL, NULL) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "w", NULL, &registers[REGISTER_W].name, NULL, NULL, NULL) );


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
        REGISTER_X, REGISTER_Y, REGISTER_Z, REGISTER_W, REGISTER_R_CTR_X
    };
    static const size_t num_requests = STACK_PTX_ARRAY_NUM_ELEMS(requests);

    static const StackPtxInstruction add_inputs[] = {
        stack_ptx_encode_routine(ROUTINE_PHILOX_PUSH_STATE),
        stack_ptx_encode_routine(ROUTINE_PHILOX_CURAND),
        stack_ptx_encode_routine(ROUTINE_PHILOX_CURAND),
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

    unsigned int add_result = 
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

    cuCheck( cuMemFree(d_out) );
    cuCheck( cuCtxDestroy(cu_context) );

    free(stack_ptx_workspace);

    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    printf("Inject (add_inputs) result (should be 8):\t%u\n", add_result);
}
