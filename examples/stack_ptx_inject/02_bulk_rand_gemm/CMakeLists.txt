project(stack_ptx_inject_02_bulk_rand_gemm)

set(PROCESSED_CUDA ${CMAKE_CURRENT_BINARY_DIR}/processed_cuda/kernel.cu)
set(ANNOTATED_PTX ${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx/kernel.ptx)
make_directory(${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx)

find_package(OpenMP REQUIRED)

add_custom_command(OUTPUT ${PROCESSED_CUDA}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu ptxinject
    COMMAND ${CMAKE_BINARY_DIR}/tools/ptxinject/ptxinject ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu -o ${CMAKE_CURRENT_BINARY_DIR}/processed_cuda -f
)

add_custom_command(OUTPUT ${ANNOTATED_PTX}
    DEPENDS ${PROCESSED_CUDA}
    COMMAND nvcc -O3 -prec-sqrt=false -arch=native -ptx ${PROCESSED_CUDA} -o ${ANNOTATED_PTX} -I ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
)

add_executable(${PROJECT_NAME} main.c)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.c OBJECT_DEPENDS ${ANNOTATED_PTX})
target_compile_definitions(${PROJECT_NAME} PRIVATE "PTX_KERNEL=${ANNOTATED_PTX}")
target_link_libraries(
    ${PROJECT_NAME} 
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    CUDA::nvJitLink
    OpenMP::OpenMP_C
    m
)
