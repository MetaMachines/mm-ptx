project(ptx_inject_00_simple)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
# add_compile_options(-pedantic-errors)

add_compile_options(
	-march=native
	-Wall
    -Werror
)

add_executable(${PROJECT_NAME} main.c)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.c OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu)
target_compile_definitions(${PROJECT_NAME} PRIVATE "CUDA_KERNEL=${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu")
target_link_libraries(
    ${PROJECT_NAME}
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
)