project(stack_ptx_inject_00_simple)

set(KERNEL_TARGET "${PROJECT_NAME}_kernel_ptx")

add_library(${KERNEL_TARGET} OBJECT kernel.cu)
target_link_libraries(${KERNEL_TARGET} PRIVATE mm_ptx_headers)
set_target_properties(${KERNEL_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_executable(${PROJECT_NAME} main.c)
add_dependencies(${PROJECT_NAME} ${KERNEL_TARGET})
target_compile_definitions(${PROJECT_NAME} PRIVATE PTX_KERNEL="$<TARGET_OBJECTS:${KERNEL_TARGET}>")
set(KERNEL_PTX "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${KERNEL_TARGET}.dir/${CMAKE_CFG_INTDIR}/kernel.ptx")
set_source_files_properties(main.c PROPERTIES OBJECT_DEPENDS "${KERNEL_PTX};${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu")

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
    stack_ptx_example_generated
)
