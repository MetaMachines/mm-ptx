#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <ptx_inject_default_generated_types.h>

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
	float x = 5;                                \n\
	float y;                                    \n\
    /* PTX_INJECT func                          \n\
        in  f32 x                               \n\
        out f32 y                               \n\
    */                                          \n\
    *out = y;                                   \n\
}                                               \n";

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

    Buffers buffers = {0};
    buffers.annotated_cuda = static_annotated_cuda;

    double elapsed[MAX_NUM_TIMING_STAGES] = {0.0};

    size_t stage = 0;

    elapsed[stage++] = clock();

    size_t num_inject_sites;
    buffers.rendered_cuda = 
    process_cuda(
        ptx_inject_data_type_infos,
        num_ptx_inject_data_type_infos,
        buffers.annotated_cuda, 
        &num_inject_sites
    );

    ASSERT( num_inject_sites == 1 );

    elapsed[stage++] = clock();

    CUdevice device;
    cuCheck( cuInit(0) );
    cuCheck( cuDeviceGet(&device, 0) );

    int device_compute_capability_major;
    int device_compute_capability_minor;

    get_device_capability(device, &device_compute_capability_major, &device_compute_capability_minor);

    buffers.annotated_ptx = nvrtc_compile(device_compute_capability_major, device_compute_capability_minor, buffers.rendered_cuda);

    elapsed[stage++] = clock();

    PtxInjectHandle ptx_inject = {0};
    ptxInjectCheck(
        ptx_inject_create(
            &ptx_inject,
            ptx_inject_data_type_infos,
            num_ptx_inject_data_type_infos,
            buffers.annotated_ptx
        )
    );

    size_t num_injects;
    ptxInjectCheck( ptx_inject_num_injects(ptx_inject, &num_injects) );

    ASSERT( num_injects == 1 );

    size_t inject_func_idx;
    ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

    PtxInjectMutType mut_type_x;
    size_t data_type_idx_x;
    const char *register_name_x;

    PtxInjectMutType mut_type_y;
    size_t data_type_idx_y;
    const char *register_name_y;

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

    ASSERT( mut_type_y == PTX_INJECT_MUT_TYPE_OUT );
    ASSERT( data_type_y == PTX_INJECT_DATA_TYPE_F32 );

    static const size_t num_ptx_stubs = 1;

    buffers.ptx_stubs = (const char **)malloc(num_ptx_stubs * sizeof(char*));
    buffers.num_ptx_stubs = num_ptx_stubs;

    char *stub_buffer = (char *)malloc(STUB_BUFFER_SIZE);
    snprintf(stub_buffer, STUB_BUFFER_SIZE, 
        "\tmul.ftz.f32 %%%1$s, %%%1$s, 0.1;\n"
        "\tadd.ftz.f32 %%%2$s, %%%1$s, 2.2;",
        register_name_x,
        register_name_y
    );

    buffers.ptx_stubs[0] = stub_buffer;

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

    elapsed[stage++] = clock();

    buffers.binary_image = 
        nvptx_compile(
            device_compute_capability_major, device_compute_capability_minor, 
            buffers.rendered_ptx, 
            rendered_ptx_bytes_written, 
            false
        );

    elapsed[stage++] = clock();

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

    printf("result: %f\n\n", h_out);
    ASSERT( h_out == 2.7f );

    cuCheck( cuMemFree(d_out) );
    cuCheck( cuModuleUnload(module) );
    cuCheck( cuCtxDestroy(context) );

    elapsed[stage++] = clock();

    for (size_t i = 1; i < stage; i++) {
        double diff = elapsed[i] - elapsed[i-1];
        diff = diff / CLOCKS_PER_SEC * 1000000.0;
        printf("Stage %d: %6d micros\n", (int)(i - 1) , (int)diff);
    }

    return 0;
}
