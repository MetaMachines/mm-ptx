project(stack_ptx_inject_06_bulk_rand_gemm_packed)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(OpenMP REQUIRED)

set(CUDA_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu)
set(REPLICATED_CUDA ${CMAKE_CURRENT_BINARY_DIR}/replicated_kernel_128.cu)

set(CUDA_REPLICATOR_SCRIPT  ${MM_PTX_TOOLS_DIR}/cuda_kernel_replicator.py CACHE FILEPATH "Path to CUDA kernel replicator script")
set(NUM_KERNEL_REPLICATIONS 16)

add_custom_command(OUTPUT ${REPLICATED_CUDA}
    DEPENDS ${CUDA_KERNEL} ${CUDA_REPLICATOR_SCRIPT}
    COMMAND ${Python3_EXECUTABLE} ${CUDA_REPLICATOR_SCRIPT}
        -n ${NUM_KERNEL_REPLICATIONS}
        -t ${CUDA_KERNEL}
        -o ${REPLICATED_CUDA}
    COMMENT "Replicating CUDA kernel ${NUM_KERNEL_REPLICATIONS} times"
)

create_injected_ptx(
    KERNEL_PTX_FILE ${REPLICATED_CUDA}
    NVCC_INCLUDE_DIRS ${MM_PTX_CUTLASS_INCLUDE_DIR}
)
add_executable(${PROJECT_NAME} main.c)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.c OBJECT_DEPENDS ${KERNEL_PTX_FILE})
target_compile_definitions(${PROJECT_NAME} PRIVATE "PTX_KERNEL=${KERNEL_PTX_FILE}")
target_compile_definitions(${PROJECT_NAME} PRIVATE "NUM_KERNEL_REPLICATIONS=${NUM_KERNEL_REPLICATIONS}")
target_link_libraries(
    ${PROJECT_NAME} 
    CUDA::cuda_driver
    # CUDA::nvrtc
    CUDA::nvptxcompiler_static
    # CUDA::nvJitLink
    OpenMP::OpenMP_C
    m
    mm_ptx_headers
)
