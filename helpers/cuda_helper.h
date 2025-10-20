#pragma once

#include <cuda.h>

#include <check_result_helper.h>

__attribute__((unused))
static
void
get_device_capability(
    CUdevice device,
    int* compute_capability_major_out,
    int* compute_capability_minor_out
) {
    cuCheck( cuDeviceGetAttribute(compute_capability_major_out, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) );
    cuCheck( cuDeviceGetAttribute(compute_capability_minor_out, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) );
}

__attribute__((unused))
static
CUresult
cuContextCreate(CUcontext* context, CUdevice device) {
    #ifdef CUDA_VERSION_13
        return cuCtxCreate(context, NULL, 0, device);
    #else
        return cuCtxCreate(context, 0, device);
    #endif
}

