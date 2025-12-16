project(stack_ptx_06_mma)

add_executable(${PROJECT_NAME} main.c)

target_link_libraries(
    ${PROJECT_NAME}
    mm_ptx_headers
    stack_ptx_example_generated
)