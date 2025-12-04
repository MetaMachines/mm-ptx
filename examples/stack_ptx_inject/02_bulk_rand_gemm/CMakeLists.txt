project(stack_ptx_inject_02_bulk_rand_gemm)

find_package(OpenMP REQUIRED)

create_injected_ptx(
    KERNEL_PTX_FILE "kernel.cu"
    NVCC_INCLUDE_DIRS ${MM_PTX_CUTLASS_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME} main.c)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.c OBJECT_DEPENDS ${KERNEL_PTX_FILE})
target_compile_definitions(${PROJECT_NAME} PRIVATE "PTX_KERNEL=\"${KERNEL_PTX_FILE}\"")
target_link_libraries(
    ${PROJECT_NAME} 
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    CUDA::nvJitLink
    OpenMP::OpenMP_C
    m
    mm_ptx_headers
)
