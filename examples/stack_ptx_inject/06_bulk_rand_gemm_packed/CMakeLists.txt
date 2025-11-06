project(stack_ptx_inject_06_bulk_rand_gemm_packed)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CUDA_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu)
set(REPLICATED_CUDA ${CMAKE_CURRENT_BINARY_DIR}/replicated_kernel_128.cu)
set(REPLICATED_PROCESSED_CUDA ${CMAKE_CURRENT_BINARY_DIR}/processed_cuda/replicated_kernel_128.cu)
set(ANNOTATED_PTX ${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx/kernel.ptx)
make_directory(${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx)

set(CUDA_REPLICATOR_SCRIPT  ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/cuda_kernel_replicator.py CACHE FILEPATH "Path to CUDA kernel replicator script")

find_package(OpenMP REQUIRED)

set(NUM_KERNEL_REPLICATIONS 16)

add_custom_command(OUTPUT ${REPLICATED_CUDA}
    DEPENDS ${CUDA_KERNEL} ${CUDA_REPLICATOR_SCRIPT}
    COMMAND ${Python3_EXECUTABLE} ${CUDA_REPLICATOR_SCRIPT}
        -n ${NUM_KERNEL_REPLICATIONS}
        -t ${CUDA_KERNEL}
        -o ${REPLICATED_CUDA}
    COMMENT "Replicating CUDA kernel ${NUM_KERNEL_REPLICATIONS} times"
)

add_custom_command(OUTPUT ${REPLICATED_PROCESSED_CUDA}
    DEPENDS ${REPLICATED_CUDA} ptxinject
    COMMAND $<TARGET_FILE:ptxinject> ${REPLICATED_CUDA} -o ${CMAKE_CURRENT_BINARY_DIR}/processed_cuda -f
)

add_custom_command(OUTPUT ${ANNOTATED_PTX}
    DEPENDS ${REPLICATED_PROCESSED_CUDA}
    COMMAND nvcc -O3 -prec-sqrt=false -arch=native -ptx ${REPLICATED_PROCESSED_CUDA} -o ${ANNOTATED_PTX} -I ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
)

add_executable(${PROJECT_NAME} main.c)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.c OBJECT_DEPENDS ${ANNOTATED_PTX})
target_compile_definitions(${PROJECT_NAME} PRIVATE "PTX_KERNEL=${ANNOTATED_PTX}")
target_compile_definitions(${PROJECT_NAME} PRIVATE "NUM_KERNEL_REPLICATIONS=${NUM_KERNEL_REPLICATIONS}")
target_link_libraries(
    ${PROJECT_NAME} 
    CUDA::cuda_driver
    # CUDA::nvrtc
    CUDA::nvptxcompiler_static
    # CUDA::nvJitLink
    OpenMP::OpenMP_C
    m
)
