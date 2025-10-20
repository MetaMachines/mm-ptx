#include <omp.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define bulkCompileCheck(ans)\
    do {\
        BulkCompileResult _bulk_result = (ans);\
        switch(_bulk_result.result_type) {\
            case BULK_COMPILE_RESULT_TYPE_SUCCESS: break;\
            case BULK_COMPILE_RESULT_TYPE_ERROR_PTX_INJECT: ptxInjectCheck( _bulk_result.code ); break;\
            case BULK_COMPILE_RESULT_TYPE_ERROR_STACK_PTX: stackPtxCheck( _bulk_result.code ); break;\
            case BULK_COMPILE_RESULT_TYPE_ERROR_OUT_OF_MEMORY: {\
                fprintf(stderr, "bulkCompileCheck: %s \n  %s %d\n", "BULK_COMPILE_RESULT_TYPE_ERROR_OUT_OF_MEMORY", __FILE__, __LINE__);    \
                assert(0);                                                                              \
                exit(1); \
                break;\
            }  \
            case BULK_COMPILE_RESULT_TYPE_RESULT_NUM_ENUMS:\
                fprintf(stderr, "bulkCompileCheck: %s \n  %s %d\n", "BULK_COMPILE_RESULT_ERROR_INVALID_ENUM", __FILE__, __LINE__);    \
                assert(0);                                                                              \
                exit(1); \
                break;\
        }\
    } while(0)

typedef struct {
    int stack_ptx_execution_limit;
    StackPtxMetaData meta_data;
    const StackPtxRequest* requests;
    int32_t num_requests;
    StackPtxInstruction** instructions;
} StackPtxStub;

typedef enum {
    BULK_COMPILE_RESULT_TYPE_SUCCESS,
    BULK_COMPILE_RESULT_TYPE_ERROR_OUT_OF_MEMORY,
    BULK_COMPILE_RESULT_TYPE_ERROR_PTX_INJECT,
    BULK_COMPILE_RESULT_TYPE_ERROR_STACK_PTX,
    BULK_COMPILE_RESULT_TYPE_RESULT_NUM_ENUMS
} BulkCompileResultType;

typedef struct {
    BulkCompileResultType result_type;
    int code;
} BulkCompileResult;

static
inline
BulkCompileResult
stack_ptx_compile_to_cubin(
    const char* nv_ptx_compile_options,
    int nv_ptx_num_compile_options,
    const char* kernel_function_name,
    const char* kernel_function_search_string,
    const char* kernel_function_replace_format_string,
    void* workspace,
    size_t workspace_size,
    size_t* workspace_bytes_written_out
) {
    
}

__attribute__((unused))
static
BulkCompileResult
stack_ptx_bulk_compile_to_cubin(
    int device_compute_capability_major,
    int device_compute_capability_minor,
    int max_num_cpu_threads,
    PtxInjectHandle ptx_inject,
    int32_t max_ast_size,
    int32_t max_ast_to_visit_stack_depth,
    int num_programs,
    StackPtxStub* stack_ptx_stubs,
    int num_stack_ptx_stubs,
    const char* kernel_function_name,
    const char* kernel_function_replace_format_string,
    void* workspace,
    size_t workspace_num_bytes,
    void* linked_cubin_buffer,
    size_t linked_cubin_buffer_num_bytes,
    size_t* max_ptx_workspace_used_out,
    size_t* max_sass_image_size_out,
    size_t* linked_cubin_buffer_bytes_written_out
) {
    BulkCompileResultType bulk_compile_result_type = BULK_COMPILE_RESULT_TYPE_SUCCESS;
    int bulk_compile_result_code = 0;
    
    // NVPTX options
    char nv_ptx_compile_line_buffer[32];
    sprintf(nv_ptx_compile_line_buffer, "--gpu-name=sm_%d%d", device_compute_capability_major, device_compute_capability_minor);
    const char* ptx_compile_options[] = {
        nv_ptx_compile_line_buffer,
        "--compile-only"
    };
    const size_t num_ptx_compile_options = 2;

    // NVJITLINK options
    char nv_jit_link_arch_option[32];
    sprintf(nv_jit_link_arch_option, "-arch=sm_%d%d", device_compute_capability_major, device_compute_capability_minor);
    const char *nv_jit_link_options[] = {
        nv_jit_link_arch_option,
        "-no-cache"
    };
    const size_t num_nv_jit_link_compile_options = 2;

    size_t ptx_workspace_idx = 0;
    char* search_string = (char*)(workspace + ptx_workspace_idx);
    ptx_workspace_idx += strlen(kernel_function_name) + 1;
    if (workspace != NULL && ptx_workspace_idx > workspace_num_bytes) {
        ASSERT( false );
    }
    if (workspace != NULL) {
        sprintf(search_string, "%s(", kernel_function_name);
    }

    // void** sass_images = (void**)malloc(num_programs * sizeof(void*));
    // size_t* sass_image_sizes = (size_t*)malloc(num_programs * sizeof(size_t));

    int omp_system_max_threads = omp_get_max_threads();
    int num_cpu_threads = max_num_cpu_threads == 0 ? omp_system_max_threads : MIN(omp_system_max_threads, max_num_cpu_threads);
#if 0
    #pragma omp parallel shared(bulk_compile_result_type, bulk_compile_result_code) num_threads(num_cpu_threads)
    {
        StackPtxResult stack_ptx_result;
        PtxInjectResult ptx_inject_result;

        StackPtxCompiler stack_ptx_compiler;
        stack_ptx_result =  
            stack_ptx_compiler_create(
                &stack_ptx_compiler, 
                0,
                stack_ptx_stubs.max_ast_to_visit_stack_depth
            );
        if (stack_ptx_result != STACK_PTX_SUCCESS) {
            #pragma omp critical  // Critical section for struct update (safer than atomic for structs in C99)
            {
                if (!bulk_compile_result_code) {  // Only set if not already set (first error wins)
                    bulk_compile_result_type = BULK_COMPILE_RESULT_TYPE_ERROR_STACK_PTX;
                    bulk_compile_result_code = stack_ptx_result;
                }
            }
        }

        char* ptx_output_buffer = (char*)malloc(stack_ptx_stubs.ptx_output_buffer_size);
        char** buffers = (char**)malloc(num_ptx_stubs_per_program * sizeof(char**));
        for (int i = 0; i < num_ptx_stubs_per_program; i++) {
            buffers[i] = (char*)malloc(stack_ptx_stubs.stack_ptx_stubs[i].stub_buffer_size);
        }

        #pragma omp for
        for (int program_idx = 0; program_idx < num_programs; program_idx++) {
            size_t num_bytes_written;
            for (int i = 0; i < num_ptx_stubs_per_program; i++) {
                StackPtxStub stack_ptx_stub = stack_ptx_stubs.stack_ptx_stubs[i];
                stackPtxCheck(
                    stack_ptx_compile(
                        stack_ptx_compiler, 
                        stack_ptx_stub.meta_data, 
                        stack_ptx_stub.instructions[program_idx],
                        stack_ptx_stub.requests, stack_ptx_stub.num_requests,
                        stack_ptx_stub.stack_ptx_execution_limit,
                        buffers[i],
                        stack_ptx_stub.stub_buffer_size,
                        &num_bytes_written
                    )
                );
            }

            ptxInjectCheck(
                ptx_inject_render_ptx(
                    ptx_inject,
                    (const char**)buffers, 
                    num_ptx_stubs_per_program,
                    ptx_output_buffer,
                    stack_ptx_stubs.ptx_output_buffer_size,
                    &num_bytes_written
                )
            );

            ptx_output_buffer[num_bytes_written-1] = '\0';

            char* start_of_name = strstr(ptx_output_buffer, search_string);
            if (start_of_name == NULL) {
                assert( false );
                exit(1);
            }
            int len = snprintf(start_of_name, strlen(search_string)+1, kernel_function_replace_format_string, program_idx);
            ASSERT( len <= strlen(kernel_function_name) );
            // sprintf in to the 00000 the number of this kernel with
            // zeros added. 
            // sprintf null terminates so we need to undo it by
            // adding the paren back.
            start_of_name[strlen(search_string)-1] = '(';

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
            // Allocate for the size of sass_image.
            sass_images[program_idx] = malloc(sass_image_size);
            // We also need the size of the sass image for nv_jit_link.
            sass_image_sizes[program_idx] = sass_image_size;
            nvptxCheck( nvPTXCompilerGetCompiledProgram(nvptx_compiler, sass_images[program_idx]) );

        }
        
        for (int i = 0; i < num_ptx_stubs_per_program; i++) {
            free(buffers[i]);
        }
        free(buffers);
        free(ptx_output_buffer);
        
        stackPtxCheck(stack_ptx_compiler_destroy(stack_ptx_compiler) );
    }

    nvJitLinkHandle nv_jit_link_handle;
    nvJitLinkCheck( nvJitLinkCreate(&nv_jit_link_handle, num_nv_jit_link_compile_options, nv_jit_link_options) );
    for (int i = 0; i < num_programs; i++) {
        nvJitLinkCheck( 
            nvJitLinkAddData(
                nv_jit_link_handle, 
                NVJITLINK_INPUT_CUBIN, 
                sass_images[num_programs - i - 1], // Going backwards makes the kernel numbers in order for the cubin.
                sass_image_sizes[num_programs - i - 1], 
                NULL
            )
        );
    }
    nvJitLinkCheck( nvJitLinkComplete(nv_jit_link_handle) );
    size_t linked_cubin_size;
    nvJitLinkCheck( nvJitLinkGetLinkedCubinSize(nv_jit_link_handle, &linked_cubin_size) );
    void* linked_cubin = malloc(linked_cubin_size);
    nvJitLinkCheck( nvJitLinkGetLinkedCubin(nv_jit_link_handle, linked_cubin) );
    nvJitLinkCheck( nvJitLinkDestroy(&nv_jit_link_handle) );

    for (int i = 0; i < num_programs; i++) {
        free(sass_images[i]);
    }
    free(sass_images);
    free(sass_image_sizes);
    free(search_string);

    *linked_cubin_out = linked_cubin;

#endif 
    BulkCompileResult bulk_compile_result;
    bulk_compile_result.result_type = bulk_compile_result_type;
    bulk_compile_result.code = bulk_compile_result_code;

    return bulk_compile_result;
}
