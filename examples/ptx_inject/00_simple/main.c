/*
 * SPDX-FileCopyrightText: 2025 MetaMachines LLC
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_DEBUG
#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_default_generated_types.h>

#include <cuda.h>
#include <helpers.h>

#define INCBIN_SILENCE_BITCODE_WARNING
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX g_
#include <incbin.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

/* Use incbin to bring the code from 00_simple.cu, allows easy editing of cuda source
*   is replaced with g_annotated_cuda_src_data
*/
INCTXT(annotated_cuda_src, XSTRING(CUDA_KERNEL));

#define STUB_BUFFER_SIZE 1000000ull

int
main() {
    printf(
        "Annotated cuda:\n"
        "---------------------------------------------\n"
        "%.*s\n"
        "---------------------------------------------\n\n",
        g_annotated_cuda_src_size,
        g_annotated_cuda_src_data
    );
    size_t num_bytes;
    size_t num_sites;

    // First measure the size we need from malloc by passing NULL for output buffer
    ptxInjectCheck(
        ptx_inject_process_cuda(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            g_annotated_cuda_src_data,
            NULL, 0,
            &num_bytes,
            &num_sites
        )
    );

    printf("Requested %zu bytes \n", num_bytes);
    printf("Found %zu injects\n\n", num_sites);
    ASSERT( num_sites == 1 );

    // Need to add 1 to the num_bytes to allow '\0' which is standard in c-libs.
    num_bytes++;

    char* processed_cuda_buffer = (char*)malloc(num_bytes);

    ptxInjectCheck(
        ptx_inject_process_cuda(
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            g_annotated_cuda_src_data,
            processed_cuda_buffer, num_bytes,
            &num_bytes,
            &num_sites
        )
    );
    printf(
        "Processed cuda:\n"
        "---------------------------------------------\n"
        "%s\n"
        "---------------------------------------------\n\n",
        processed_cuda_buffer
    );

    // We now need the device capability before we can compile with nvrtc

    CUdevice device;
    cuCheck( cuInit(0) );
    cuCheck( cuDeviceGet(&device, 0) );

    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(device, &device_compute_capability_major, &device_compute_capability_minor);

    printf("Device(0) has compute capability: sm_%d%d\n\n", device_compute_capability_major, device_compute_capability_minor);

    // Now we can compile using the nvrtc helper
    char* compiled_ptx = nvrtc_compile(device_compute_capability_major, device_compute_capability_minor, processed_cuda_buffer);

    // Can free processed_cuda_buffer
    free(processed_cuda_buffer);

    printf(
        "Compiled ptx:\n"
        "---------------------------------------------\n"
        "%s\n"
        "---------------------------------------------\n\n",
        compiled_ptx
    );

    // Now we can initialize a PtxInjectHandle
    PtxInjectHandle ptx_inject;
    ptxInjectCheck( 
        ptx_inject_create(
            &ptx_inject, 
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            compiled_ptx
        )
    );
    
    // We can now free the compiled_ptx buffer
    free(compiled_ptx);

    size_t num_injects_found;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects_found) );

    // Print the data we found in the compiled ptx
    print_ptx_inject_info(ptx_inject, ptx_inject_data_type_infos);
    printf("\n");

    // We can now query the ptx_inject handle for meta data or we just "know" what
    // the inject and variable names are.

    const char* register_name_x;
    const char* register_name_y;
    const char* register_name_z;

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    // We can pass NULL for any field we don't care about. i.e. mut_type and data_type
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

    // Grab the size of the necessary buffer to hold the rendered_ptx
    ptxInjectCheck(
        ptx_inject_render_ptx(
            ptx_inject, ptx_stubs, 1, NULL, 0, &num_bytes
        )
    );

    // increment size again to account for the null terminator
    num_bytes++;

    char* rendered_ptx = (char*)malloc(num_bytes);
    ptxInjectCheck(
        ptx_inject_render_ptx(
            ptx_inject, ptx_stubs, 1, rendered_ptx, num_bytes, &num_bytes
        )
    );

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

    // We can now compile this ptx to sass
    void* sass = nvptx_compile(device_compute_capability_major, device_compute_capability_minor, rendered_ptx, num_bytes, false);

    // Free rendered_ptx buffer
    free(rendered_ptx);

    // Now let's run this kernel!
    CUcontext context;
    cuCheck( cuContextCreate(&context, device) );

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
    cuCheck( cuCtxDestroy(context) );
}
