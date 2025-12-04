project(stack_ptx_inject_05_bulk_simple_packed)

find_package(OpenMP REQUIRED)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CUDA_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu)
set(REPLICATED_CUDA ${CMAKE_CURRENT_BINARY_DIR}/replicated_kernel_128.cu)

set(CUDA_REPLICATOR_SCRIPT  ${MM_PTX_TOOLS_DIR}/cuda_kernel_replicator.py CACHE FILEPATH "Path to CUDA kernel replicator script")

add_custom_command(OUTPUT ${REPLICATED_CUDA}
    DEPENDS ${CUDA_KERNEL} ${CUDA_REPLICATOR_SCRIPT}
    COMMAND ${Python3_EXECUTABLE} ${CUDA_REPLICATOR_SCRIPT}
        -n 128
        -t ${CUDA_KERNEL}
        -o ${REPLICATED_CUDA}
    COMMENT "Replicating CUDA kernel 128 times"
)

create_injected_ptx(
    KERNEL_PTX_FILE ${REPLICATED_CUDA}
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
