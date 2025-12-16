project(stack_ptx_08_cpp)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(
    ${PROJECT_NAME}
    mm_ptx_headers
    stack_ptx_example_generated
)