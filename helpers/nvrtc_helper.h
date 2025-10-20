#pragma once

#include <nvrtc.h>

#include <check_result_helper.h>

static
void
nvrtc_print_info_log(
    nvrtcProgram nvptx_program
) {
    size_t info_size;
    nvrtcCheck(
        nvrtcGetProgramLogSize(nvptx_program, &info_size)
    );

    if (info_size != 0) {
        char *info_log = (char*)malloc(info_size+1);
        nvrtcCheck(nvrtcGetProgramLog(nvptx_program, info_log));
        printf("Error log: %s\n", info_log);
        free(info_log);
    }
}

char *
nvrtc_compile(
    int compute_capability_major,
    int compute_capability_minor,
    const char *cuda_code
) {
    char compile_line_buffer[1024];
    sprintf(compile_line_buffer, "-arch=sm_%d%d", compute_capability_major, compute_capability_minor);

    const char* cuda_compile_options[] = {
        compile_line_buffer
    };
    const size_t num_cuda_compile_options = 1;

    nvrtcProgram program = {0};
    nvrtcCheck(
        nvrtcCreateProgram(
            &program,
            cuda_code,
            NULL,
            0,
            NULL,
            NULL
        )
    );

    nvrtcResult nvrtc_result = 
        nvrtcCompileProgram(
            program, num_cuda_compile_options, cuda_compile_options
        );
    if (nvrtc_result != NVRTC_SUCCESS) {
        nvrtc_print_info_log(program);
    }

    size_t ptx_src_size;
    nvrtcCheck(
        nvrtcGetPTXSize(program, &ptx_src_size)
    );

    char *ptx_src = (char*)malloc(ptx_src_size);
    nvrtcCheck(
        nvrtcGetPTX(program, ptx_src)
    );

    nvrtcCheck(
        nvrtcDestroyProgram(&program)
    );

    return ptx_src;
}
