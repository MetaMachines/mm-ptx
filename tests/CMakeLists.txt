set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_subdirectory(ptx_inject_full)
add_subdirectory(stack_ptx)
add_subdirectory(stack_ptx_inject_full)
