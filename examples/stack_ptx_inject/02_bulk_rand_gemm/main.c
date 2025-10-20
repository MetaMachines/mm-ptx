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

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

// Use incbin to drag the kernel.ptx compiled from kernel.cu into this
// compilation unit as a static string. kernel.cu was already processed by
// the ptxinject binary and compiled with nvcc -ptx.
INCBIN(char, annotated_ptx, XSTRING(PTX_KERNEL));

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// We have three sites in the Cutlass/CuTe gemm that we can replace:
//  * multiply operation injected as:
    /* PTX_INJECT multiply
        in  f32 v_a
        in  f32 v_b
        out f32 diff 
    */
//  * accumulate operation injected as:
    /* PTX_INJECT accumulate
        in  f32 v_c
        in  f32 diff
        out f32 v_c_out
    */
//  * epilogue injected as:
    /* PTX_INJECT epilogue
        mod f32 v_c
    */
// We will do:
//  * multiply as: diff = binary(unary(v_a), unary(v_b))
//  * accumulate as: v_c_out = binary(diff, v_c)
//  * epilogue as: v_c = unary(v_c)

// Struct to hold the instructions for the above gemm kernel.
typedef struct {
    StackPtxInstruction multiply_unary_v_a;
    StackPtxInstruction multiply_unary_v_b;
    StackPtxInstruction multiply_binary_v_a_v_b;
    StackPtxInstruction accumulate_binary_diff_v_c;
    StackPtxInstruction epilogue_unary_v_c;
} InstructionLayout;

#define _ALIGNMENT 16 // Standard malloc alignment
#define _ALIGNMENT_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))

#define NUM_RAND_GEMM_KERNELS           2048
// MAX_NUM_CPU_THREADS 0 will mean we'll use all threads found.
#define MAX_NUM_CPU_THREADS             0

#define STACK_PTX_EXECUTION_LIMIT       100

// workspace (per-thread) for StackPtx, PTX stubs and output PTX
#define WORKSPACE_BUFFER_SIZE           (1ULL << 20)
// SASS image storage for output from nvPTX, 100KiB per SASS.
#define SASS_IMAGE_STORAGE              (100ULL * (1ULL << 10))

// Prints the InstructionLayout and the related matrix
#define MATRIX_PRINT_RESULTS            0
// Number of rows and columns to print of the matrix
#define MATRIX_PRINT_LIMIT              5

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

int
main() {
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

    // Now we can populate the instruction layouts for each random gemm kernel
    InstructionLayout* instruction_layouts = (InstructionLayout*)malloc(NUM_RAND_GEMM_KERNELS * sizeof(InstructionLayout));
    for (int i = 0; i < NUM_RAND_GEMM_KERNELS; i++) {
        InstructionLayout* instruction_layout = &instruction_layouts[i];
        instruction_layout->multiply_unary_v_a =            unary_f32_instructions[rand() % num_unary_f32_instructions];
        instruction_layout->multiply_unary_v_b =            unary_f32_instructions[rand() % num_unary_f32_instructions];
        instruction_layout->multiply_binary_v_a_v_b =       binary_f32_instructions[rand() % num_binary_f32_instructions];
        instruction_layout->accumulate_binary_diff_v_c =    binary_f32_instructions[rand() % num_binary_f32_instructions];
        instruction_layout->epilogue_unary_v_c =            unary_f32_instructions[rand() % num_unary_f32_instructions];
    }

    // Now we create the PtxInjectHandle loading in the annotated ptx.
    // This will prepare the PTX to inject the random gemm kernel stubs.
    PtxInjectHandle ptx_inject;
    ptxInjectCheck(
        ptx_inject_create(
            &ptx_inject, 
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            g_annotated_ptx_data
        )
    );

    // We'll print the info it found with this helper function.
    printf("Inject Info:\n");
    print_ptx_inject_info(ptx_inject, ptx_inject_data_type_infos);
    printf("\n");

    // Our kernel has three injects in the CUDA file. We'll make sure
    // the PTX Inject system found three unique injects.
    size_t num_injects;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects) );

    ASSERT( num_injects == 3 );

    size_t multiply_func_idx;
    size_t accumulate_func_idx;
    size_t epilogue_func_idx;

    size_t multiply_num_args;
    size_t multiply_num_sites;
    ptxInjectCheck( 
        ptx_inject_inject_info_by_name(ptx_inject, "multiply", &multiply_func_idx, &multiply_num_args, &multiply_num_sites)
    );

    size_t accumulate_num_args;
    size_t accumulate_num_sites;
    ptxInjectCheck( 
        ptx_inject_inject_info_by_name(ptx_inject, "accumulate", &accumulate_func_idx, &accumulate_num_args, &accumulate_num_sites)
    );

    size_t epilogue_num_args;
    size_t epilogue_num_sites;
    ptxInjectCheck( 
        ptx_inject_inject_info_by_name(ptx_inject, "epilogue", &epilogue_func_idx, &epilogue_num_args, &epilogue_num_sites) 
    );
    
    // Another sanity check for the kernel.
    ASSERT( multiply_num_args == 3 );
    ASSERT( accumulate_num_args == 3 );
    ASSERT( epilogue_num_args == 1 );

    // Now we'll grab the normalized PTX register names.
    enum Registers {
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
        [REGISTER_MULTIPLY_OUTPUT_DIFF] = { .name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_MULTIPLY_INPUT_V_A] = { .name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_MULTIPLY_INPUT_V_B] = { .name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_ACCUMULATE_OUTPUT_V_C_OUT] = { .name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_ACCUMULATE_INPUT_V_C] = { .name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_ACCUMULATE_INPUT_DIFF] = { .name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
        [REGISTER_EPILOGUE_MODIFY_V_C] = { .name = NULL, .stack_idx = STACK_PTX_STACK_TYPE_F32 },
    };

    // We'll now grab the actual register names. The NULL fields are variable info fields that we won't check at this time like the MutType, DataType and arg index.
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "diff", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_OUTPUT_DIFF].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "v_a", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_INPUT_V_A].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, multiply_func_idx, "v_b", NULL, NULL, NULL, &registers[REGISTER_MULTIPLY_INPUT_V_B].name) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "v_c_out", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_OUTPUT_V_C_OUT].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "v_c", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_INPUT_V_C].name) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, accumulate_func_idx, "diff", NULL, NULL, NULL, &registers[REGISTER_ACCUMULATE_INPUT_DIFF].name) );

    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, epilogue_func_idx, "v_c", NULL, NULL, NULL, &registers[REGISTER_EPILOGUE_MODIFY_V_C].name) );

    // The requests will stay the same also. These are requests for the instructions ran to
    // be assigned to these register names.

    // We need the multiply instructions to be assigned to the "diff" register name
    const size_t multiply_requests[] = { REGISTER_MULTIPLY_OUTPUT_DIFF };
    static const size_t num_multiply_requests = STACK_PTX_ARRAY_NUM_ELEMS(multiply_requests);

    // We need the accumulate instructions to be assigned to the "v_c_out" register name
    const size_t accumulate_requests[] = { REGISTER_ACCUMULATE_OUTPUT_V_C_OUT };
    static const size_t num_accumulate_requests = STACK_PTX_ARRAY_NUM_ELEMS(accumulate_requests);

    // We need the epilogue instructions to be assigned to the "v_c" register name.
    // This register has a modify type so it is used now as an output and later as an input as well.
    const size_t epilogue_requests[] = { REGISTER_EPILOGUE_MODIFY_V_C };
    static const size_t num_epilogue_requests = STACK_PTX_ARRAY_NUM_ELEMS(epilogue_requests);

    // Now we'll initialize the CUDA driver API and grab the compute capability of the first device.
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
        "--compile-only"
    };
    const size_t num_ptx_compile_options = 2;

    size_t omp_system_max_threads = omp_get_max_threads();
    int num_cpu_threads = MAX_NUM_CPU_THREADS == 0 ? omp_system_max_threads : MIN(omp_system_max_threads, MAX_NUM_CPU_THREADS);
    printf(
        "CPU threads (%d found, %d using)\n",
        (int)omp_system_max_threads,
        num_cpu_threads
    );

    // Going to calculate sizes and offsets for slicing up the single malloc call in to their respective buffers and workspaces.
    size_t linked_cubin_image_num_bytes = NUM_RAND_GEMM_KERNELS * SASS_IMAGE_STORAGE;
    size_t sass_image_num_bytes = NUM_RAND_GEMM_KERNELS * SASS_IMAGE_STORAGE;
    size_t sass_image_sizes_num_bytes = NUM_RAND_GEMM_KERNELS * sizeof(size_t);
    size_t workspace_num_bytes_num_bytes = num_cpu_threads * WORKSPACE_BUFFER_SIZE;

    size_t linked_cubin_image_offset = 0;
    size_t sass_image_offset = linked_cubin_image_offset + _ALIGNMENT_UP(linked_cubin_image_num_bytes, _ALIGNMENT);
    size_t sass_image_sizes_offset = sass_image_offset + _ALIGNMENT_UP(sass_image_num_bytes, _ALIGNMENT);
    size_t workspace_offset = sass_image_sizes_offset + _ALIGNMENT_UP(sass_image_sizes_num_bytes, _ALIGNMENT);
    size_t total_num_bytes = workspace_offset + workspace_num_bytes_num_bytes;

    printf(
        "Using %zu MiB of memory for workspaces, sass buffers and final cubin\n"
        "\n",
        total_num_bytes / 1024 / 1024
    );

    void* memory_block = (void*)malloc(total_num_bytes);
    ASSERT( memory_block != NULL );

    // We'll create the pointers that will store the sass binary images for all NUM_RAND_GEMM_KERNELS
    void* linked_cubin = (void*)(memory_block + linked_cubin_image_offset);
    void* sass_images = (void*)(memory_block + sass_image_offset);
    size_t* sass_image_sizes = (size_t*)(memory_block + sass_image_sizes_offset);

    double time_start, time_end;

    enum PASSES {
        PASS_PTX_ONLY = 0,
        PASS_PTX_AND_SASS = 1,
        PASSES_NUM_ENUMS
    };

    size_t stack_ptx_workspace_size;
    stackPtxCheck(
        stack_ptx_compile_workspace_size(
            &compiler_info,
            &stack_ptx_stack_info,
            &stack_ptx_workspace_size
        )
    );

    // We're going to store a WORKPACE_BUFFER_SIZE amount of bytes per thread.
    void* workspace_buffers = (void*)(memory_block + workspace_offset);

    size_t max_ptx_workspace_bytes = 0;

    for (int pass = 0; pass < PASSES_NUM_ENUMS; pass++) {
        time_start = omp_get_wtime();
        #pragma omp parallel num_threads(num_cpu_threads)
        {
            // We'll get each threads respective buffer.
            void* workspace_buffer = workspace_buffers + omp_get_thread_num() * WORKSPACE_BUFFER_SIZE;
            // And track it's offset as its used for various purposes.
            size_t current_workspace_buffer_idx = 0;

            #pragma omp for
            for (int i = 0; i < NUM_RAND_GEMM_KERNELS; i++) {
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

                size_t num_bytes_written;
                // We need to start it off past the workspace buffer for stack_ptx_compile.
                current_workspace_buffer_idx = stack_ptx_workspace_size;
                // This buffer starts at the end of the previous buffer.
                char* multiply_stub_buffer = workspace_buffer + current_workspace_buffer_idx;
                stackPtxCheck(
                    stack_ptx_compile(
                        &compiler_info,
                        &stack_ptx_stack_info,
                        multiply_instructions,
                        registers, REGISTER_NUM_ENUMS,
                        NULL, 0,
                        multiply_requests,
                        num_multiply_requests,
                        STACK_PTX_EXECUTION_LIMIT,
                        workspace_buffer,
                        stack_ptx_workspace_size,
                        multiply_stub_buffer,
                        WORKSPACE_BUFFER_SIZE - current_workspace_buffer_idx,
                        &num_bytes_written
                    )
                );
                // Need to add one due to null terminator.
                current_workspace_buffer_idx += num_bytes_written + 1;
                // This buffer starts at the end of the previous buffer.
                char* accumulate_stub_buffer = workspace_buffer + current_workspace_buffer_idx;
                stackPtxCheck(
                    stack_ptx_compile(
                        &compiler_info,
                        &stack_ptx_stack_info,
                        accumulate_instructions,
                        registers, REGISTER_NUM_ENUMS,
                        NULL, 0,
                        accumulate_requests,
                        num_accumulate_requests,
                        STACK_PTX_EXECUTION_LIMIT,
                        workspace_buffer,
                        stack_ptx_workspace_size,
                        accumulate_stub_buffer,
                        WORKSPACE_BUFFER_SIZE - current_workspace_buffer_idx,
                        &num_bytes_written
                    )
                );
                current_workspace_buffer_idx += num_bytes_written + 1;
                // This buffer starts at the end of the previous buffer.
                char* epilogue_stub_buffer = workspace_buffer + current_workspace_buffer_idx;
                stackPtxCheck(
                    stack_ptx_compile(
                        &compiler_info,
                        &stack_ptx_stack_info,
                        epilogue_instructions,
                        registers, REGISTER_NUM_ENUMS,
                        NULL, 0,
                        epilogue_requests,
                        num_epilogue_requests,
                        STACK_PTX_EXECUTION_LIMIT,
                        workspace_buffer,
                        stack_ptx_workspace_size,
                        epilogue_stub_buffer,
                        WORKSPACE_BUFFER_SIZE - current_workspace_buffer_idx,
                        &num_bytes_written
                    )
                );
                current_workspace_buffer_idx += num_bytes_written + 1;
                // This buffer starts at the end of the previous buffer.
                char* ptx_output_buffer = workspace_buffer + current_workspace_buffer_idx;

                const char* ptx_stubs[3];
                ptx_stubs[multiply_func_idx] = multiply_stub_buffer;
                ptx_stubs[accumulate_func_idx] = accumulate_stub_buffer;
                ptx_stubs[epilogue_func_idx] = epilogue_stub_buffer;

                ptxInjectCheck(
                    ptx_inject_render_ptx(
                        ptx_inject,
                        ptx_stubs, 3,
                        ptx_output_buffer,
                        WORKSPACE_BUFFER_SIZE - current_workspace_buffer_idx,
                        &num_bytes_written
                    )
                );
                max_ptx_workspace_bytes = STACK_PTX_MAX(max_ptx_workspace_bytes, num_bytes_written);
                // For posterity if anything else would follow.
                current_workspace_buffer_idx += num_bytes_written + 1;

                // INCBIN brings in the ptx without a null terminator so we'll terminate it here
                ptx_output_buffer[num_bytes_written-1] = '\0';

                // We also need to rename the kernel in this ptx so we can
                // merge all of the sass into one cubin file. We named the 
                // kernel "gemm_nt_00000" with extern "C", so we should be able to
                // find it easily. If we can't find it then we should crash.
                char* start_of_name = strstr(ptx_output_buffer, "gemm_nt_00000(");
                if (start_of_name == NULL) {
                    assert( false );
                    exit(1);
                }
                // sprintf in to the 00000 the number of this kernel with
                // zeros added.
                sprintf(start_of_name, "gemm_nt_%05d", i);
                // sprintf null terminates so we need to undo it by
                // adding the paren back.
                start_of_name[13] = '(';

                if (pass == PASS_PTX_ONLY) {
                    continue;
                }

                nvPTXCompilerHandle nvptx_compiler;
                nvptxCheck(
                    nvPTXCompilerCreate(
                        &nvptx_compiler,
                        num_bytes_written,
                        ptx_output_buffer
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
                nvptxCheck( nvPTXCompilerGetCompiledProgramSize(nvptx_compiler, &sass_image_size) );
                ASSERT( sass_image_size <= SASS_IMAGE_STORAGE );
                // Allocate for the size of sass_image.
                void* sass_image = sass_images + i * SASS_IMAGE_STORAGE;
                // We also need the size of the sass image for nv_jit_link.
                sass_image_sizes[i] = sass_image_size;
                nvptxCheck( nvPTXCompilerGetCompiledProgram(nvptx_compiler, sass_image) );
            }
        }
        time_end = omp_get_wtime();
        
        if (pass == PASS_PTX_ONLY) {
            printf("PTX generation (StackPTX + PtxInject - nvPTXCompiler)\n");
        } else if (pass == PASS_PTX_AND_SASS) {
            printf("SASS compilation (StackPTX + PtxInject + nvPTXCompiler)\n");
        }
        printf(
            "  (%d kernels, %d cpu threads):\n"
            "\t%f seconds\n"
            "\t%0.0f kernels/second\n"
            "\t%0.3f microseconds/kernel\n",
            NUM_RAND_GEMM_KERNELS,
            num_cpu_threads,
            (time_end - time_start),
            (double)NUM_RAND_GEMM_KERNELS / (time_end - time_start),
            (time_end - time_start) / (double)NUM_RAND_GEMM_KERNELS * (double)1000000
        );
    }

    // We don't need the PTX Inject handle anymore.
    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    // We'll now setup nvJitLink to merge all the sass in to one kernel.
    char nv_jit_link_arch_option[32];
    sprintf(nv_jit_link_arch_option, "-arch=sm_%d%d", device_compute_capability_major, device_compute_capability_minor);
    const char *nv_jit_link_options[] = {
        nv_jit_link_arch_option,
        "-no-cache"
    };
    const size_t num_nv_jit_link_compile_options = 2;

    time_start = omp_get_wtime();
    nvJitLinkHandle nv_jit_link_handle;
    nvJitLinkCheck( nvJitLinkCreate(&nv_jit_link_handle, num_nv_jit_link_compile_options, nv_jit_link_options) );
    for (int i = 0; i < NUM_RAND_GEMM_KERNELS; i++) {
        // Going backwards makes the kernel numbers in order for the cubin.
        void* sass_image = sass_images + (size_t)(NUM_RAND_GEMM_KERNELS - i -1) * SASS_IMAGE_STORAGE;
        nvJitLinkCheck( 
            nvJitLinkAddData(
                nv_jit_link_handle, 
                NVJITLINK_INPUT_CUBIN, 
                sass_image, 
                sass_image_sizes[NUM_RAND_GEMM_KERNELS - i - 1], 
                NULL
            )
        );
    }
    nvJitLinkCheck( nvJitLinkComplete(nv_jit_link_handle) );
    size_t linked_cubin_size;
    nvJitLinkCheck( nvJitLinkGetLinkedCubinSize(nv_jit_link_handle, &linked_cubin_size) );
    ASSERT( linked_cubin_image_offset <= linked_cubin_image_num_bytes );
    nvJitLinkCheck( nvJitLinkGetLinkedCubin(nv_jit_link_handle, linked_cubin) );
    nvJitLinkCheck( nvJitLinkDestroy(&nv_jit_link_handle) );
    time_end = omp_get_wtime();
    printf(
        "nvJitLink (1 cpu thread, %d kernels): %f seconds\n"
        "\n",
        NUM_RAND_GEMM_KERNELS,
        (time_end - time_start)
    );

    // Now we can run these kernels!
    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, cu_device) );

    CUmodule module;
    cuCheck( cuModuleLoadDataEx(&module, linked_cubin, 0, NULL, NULL) );

    unsigned int function_count;
    cuCheck( cuModuleGetFunctionCount(&function_count, module) );

    ASSERT( function_count == NUM_RAND_GEMM_KERNELS );
    CUfunction* functions = (CUfunction*)malloc(function_count * sizeof(CUfunction));

    cuCheck( cuModuleEnumerateFunctions(functions, function_count, module) );

    // Now we'll set up the gemm tensors. The tensor size will stay small
    // and aligned to 128 because the CuTe/Cutlass tensor in its current
    // form doesn't support predication. We'll launch one block per
    // kernel.
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
    cuCheck( cuMemAlloc(&d_cs, d_c_matrix_size * NUM_RAND_GEMM_KERNELS) );

    cuCheck( cuMemcpyHtoD(d_a, h_a, (size_t)M * (size_t)K * sizeof(float)) );
    cuCheck( cuMemcpyHtoD(d_b, h_b, (size_t)N * (size_t)K * sizeof(float)) );

    CUstream stream;
    cuCheck( cuStreamCreate(&stream, 0) );

    time_start = omp_get_wtime();
    for (int i = 0; i < NUM_RAND_GEMM_KERNELS; i++) {
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
        NUM_RAND_GEMM_KERNELS,
        (time_end - time_start)
    );

    if (MATRIX_PRINT_RESULTS) {
        for (int i = 0; i < NUM_RAND_GEMM_KERNELS; i++) {
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

    free(instruction_layouts);
    free(memory_block);
}
