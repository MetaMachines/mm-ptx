/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_default_generated_types.h>
#include <ptx_inject_helper.h>

#include <cuda.h>
#include <cuda_helper.h>
#include <nvptx_helper.h>

#include <check_result_helper.h>
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

int
main() {
    printf("Annotated PTX:\n"
        "---------------------------------------------\n"
        "%.*s"
        "---------------------------------------------\n\n",
         g_annotated_ptx_size, g_annotated_ptx_data
    );

    // The cmake plumbing already used the ptxinject cli tool compiled inside the 
    // project to process kernel.cu. The cuda was then compiled by nvcc as part of
    // the cmake process as well. INCBIN added the ptx to this file as g_annotated_ptx_data.

    PtxInjectHandle ptx_inject;
    ptxInjectCheck( 
        ptx_inject_create(
            &ptx_inject, 
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            g_annotated_ptx_data
        )
    );

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    const char* register_name_x;
    const char* register_name_y;
    const char* register_name_z;
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "x", NULL, NULL, NULL, &register_name_x) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "y", NULL, NULL, NULL, &register_name_y) );
    ptxInjectCheck( ptx_inject_variable_info_by_name(ptx_inject, inject_func_idx, "z", NULL, NULL, NULL, &register_name_z) );

     // We will inject a simple add instruction that just does z = y + x;
    char *stub_buffer = (char *)malloc(STUB_BUFFER_SIZE);
    snprintf(stub_buffer, STUB_BUFFER_SIZE, 
        "\tadd.ftz.f32 %%%3$s, %%%2$s, %%%1$s;",
        register_name_x,
        register_name_y,
        register_name_z
    );

    // If there are more than one injects in the ptx, then we need to know which order to send the
    // stubs to 'ptx_inject_render_ptx. In this case because there is only one, 'func' will
    // always be at index 0.

    

    const char* ptx_stubs[1];
    ptx_stubs[inject_func_idx] = stub_buffer;

    // We'll use the local helper this time
    size_t num_bytes_written;
    char* rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 1, &num_bytes_written);

    // We should now see the add instruction inside the ptx.
    printf(
        "Rendered ptx:\n"
        "---------------------------------------------\n"
        "%s\n"
        "---------------------------------------------\n\n",
        rendered_ptx
    );

    // We can now destroy the PtxInjectHandle
    ptxInjectCheck( ptx_inject_destroy(ptx_inject) );

    cuCheck( cuInit(0) );
    CUdevice device;
    
    cuCheck( cuDeviceGet(&device, 0) );
    
    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    // We can now compile this ptx to sass
    void* sass = nvptx_compile(device_compute_capability_major, device_compute_capability_minor, rendered_ptx, num_bytes_written, NULL, false);

    // Free rendered_ptx buffer
    free(rendered_ptx);

    // Now let's run this kernel!
    CUcontext cu_context;
    cuCheck( cuContextCreate(&cu_context, device) );

    CUdeviceptr d_out;
    cuCheck( cuMemAlloc(&d_out, sizeof(float)) );

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

    float h_out;
    cuCheck( cuMemcpyDtoH(&h_out, d_out, sizeof(float)) );

    // For this given example, the result should be 8
    printf("Result (should be 8): %f\n", h_out);

    cuCheck( cuModuleUnload(cu_module) );

    cuCheck( cuMemFree(d_out) );
    cuCheck( cuCtxDestroy(cu_context) );
}
