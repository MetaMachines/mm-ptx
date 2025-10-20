#pragma once

#include <nvPTXCompiler.h>

#include <check_result_helper.h>

static
void
nvptx_print_info_log(
    nvPTXCompilerHandle nvptx_compiler
) {
    size_t info_size;
    nvptxCheck(
        nvPTXCompilerGetInfoLogSize(nvptx_compiler, &info_size)
    );

    if (info_size != 0) {
        char *info_log = (char*)malloc(info_size+1);
        nvptxCheck(nvPTXCompilerGetInfoLog(nvptx_compiler, info_log));
        printf("Error log: %s\n", info_log);
        free(info_log);
    }
}

static
void
nvptx_print_error_log(
    nvPTXCompilerHandle nvptx_compiler
) {
    size_t error_size;
    nvptxCheck(
        nvPTXCompilerGetErrorLogSize(nvptx_compiler, &error_size)
    );

    if (error_size != 0) {
        char *error_log = (char*)malloc(error_size+1);
        nvptxCheck(nvPTXCompilerGetErrorLog(nvptx_compiler, error_log));
        fprintf(stderr, "Error log: %s\n", error_log);
        free(error_log);
    }
}

__attribute__((unused))
static
char *
nvptx_compile(
    int compute_capability_major,
    int compute_capability_minor,
    const char *ptx_code,
    size_t ptx_code_size,
    bool verbose
) {
    char compile_line_buffer[32];
    sprintf(compile_line_buffer, "--gpu-name=sm_%d%d", compute_capability_major, compute_capability_minor);

    const char* ptx_compile_options[] = {
        compile_line_buffer,
        verbose ? "--verbose" : NULL
    };
    const size_t num_ptx_compile_options = verbose ? 2 : 1;

    nvPTXCompilerHandle nvptx_compiler = {0};
    nvptxCheck(
        nvPTXCompilerCreate(
            &nvptx_compiler,
            ptx_code_size,
            ptx_code
        )
    );

    nvPTXCompileResult result = 
        nvPTXCompilerCompile(
            nvptx_compiler,
            num_ptx_compile_options,
            ptx_compile_options
        );

    nvptx_print_info_log(nvptx_compiler);

    if (result != NVPTXCOMPILE_SUCCESS) {
        nvptx_print_error_log(nvptx_compiler);
        assert( false );
        exit(1);
    }

    size_t binary_image_size;
    nvptxCheck(
        nvPTXCompilerGetCompiledProgramSize(
            nvptx_compiler, &binary_image_size
        )
    );

    char *binary_image = (char*)malloc(binary_image_size);
    nvptxCheck(
        nvPTXCompilerGetCompiledProgram(
            nvptx_compiler, binary_image
        )
    );
    nvptxCheck(
        nvPTXCompilerDestroy(&nvptx_compiler)
    );

    return binary_image;
}
