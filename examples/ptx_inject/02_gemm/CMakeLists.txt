project(ptx_inject_02_gemm)

set(PROCESSED_CUDA ${CMAKE_CURRENT_BINARY_DIR}/processed_cuda/kernel.cu)
set(ANNOTATED_PTX ${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx/kernel.ptx)
make_directory(${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx)

add_custom_command(OUTPUT ${PROCESSED_CUDA}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu ptxinject
    COMMAND $<TARGET_FILE:ptxinject> ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu -o ${CMAKE_CURRENT_BINARY_DIR}/processed_cuda -f
)

add_custom_command(OUTPUT ${ANNOTATED_PTX}
    DEPENDS ${PROCESSED_CUDA}
    COMMAND nvcc -O3 -prec-sqrt=false -arch=native -ptx ${PROCESSED_CUDA} -o ${ANNOTATED_PTX} -I ${MM_PTX_CUTLASS_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME} main.c)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.c OBJECT_DEPENDS ${ANNOTATED_PTX})
target_compile_definitions(${PROJECT_NAME} PRIVATE "PTX_KERNEL=${ANNOTATED_PTX}")
target_link_libraries(
    ${PROJECT_NAME}
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
)
