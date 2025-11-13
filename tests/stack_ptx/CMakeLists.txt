project(test_stack_ptx C)

# add_compile_options(-pedantic-errors)

add_executable(
    ${PROJECT_NAME}
    main.c
)

target_link_libraries(
    ${PROJECT_NAME}
    mm_ptx_headers
)

