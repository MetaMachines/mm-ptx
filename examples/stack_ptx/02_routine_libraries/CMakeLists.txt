project(stack_ptx_02_routine_libraries)

include_directories(.)

add_executable(${PROJECT_NAME} main.c)

target_link_libraries(
    ${PROJECT_NAME}
    mm_ptx_headers
    stack_ptx_example_generated
)