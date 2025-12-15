cmake_minimum_required(VERSION 3.18)
project(ptx_inject_new_00_simple LANGUAGES C CUDA)

add_library(kernel_ptx OBJECT kernel.cu)
target_link_libraries(kernel_ptx PRIVATE mm_ptx_headers)
set_target_properties(kernel_ptx PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_executable(${PROJECT_NAME} main.c)
add_dependencies(${PROJECT_NAME} kernel_ptx)
target_compile_definitions(${PROJECT_NAME} PRIVATE PTX_KERNEL="$<TARGET_OBJECTS:kernel_ptx>")
set_source_files_properties(main.c PROPERTIES OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu")

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
)
