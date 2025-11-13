project(stack_ptx_07_store_load)

add_executable(${PROJECT_NAME} main.c)

target_link_libraries(
    ${PROJECT_NAME}
    mm_ptx_headers
)
