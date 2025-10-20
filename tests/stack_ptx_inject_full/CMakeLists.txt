project(test_stack_ptx_inject_full C)

add_compile_options(
	-march=native
	-Wall
    -Wextra
)

add_executable(${PROJECT_NAME} main.c)

target_link_libraries(
    ${PROJECT_NAME}
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    CUDA::cuda_driver
    m
)
