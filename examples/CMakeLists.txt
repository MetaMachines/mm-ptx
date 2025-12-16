
set(GENERATED_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(STACK_PTX_DESCRIPTIONS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/type_descriptions/stack_ptx_descriptions.json")
set(STACK_PTX_GENERATED_C_HDR    "${GENERATED_INC_DIR}/stack_ptx_example_descriptions.h")
set(STACK_PTX_GENERATED_CPP_HDR    "${GENERATED_INC_DIR}/stack_ptx_example_descriptions.hpp")

mm_add_stack_ptx_header(
    OUT_HDR "${STACK_PTX_GENERATED_C_HDR}"
    JSON    "${STACK_PTX_DESCRIPTIONS_JSON}"
    LANG    c
)

mm_add_stack_ptx_header(
    OUT_HDR "${STACK_PTX_GENERATED_CPP_HDR}"
    JSON    "${STACK_PTX_DESCRIPTIONS_JSON}"
    LANG    cpp
)

add_custom_target(stack_ptx_example_generated_headers
  DEPENDS
    "${STACK_PTX_GENERATED_C_HDR}"
    "${STACK_PTX_GENERATED_CPP_HDR}"
)

add_library(stack_ptx_example_generated INTERFACE)
target_include_directories(stack_ptx_example_generated INTERFACE "${GENERATED_INC_DIR}")
add_dependencies(stack_ptx_example_generated stack_ptx_example_generated_headers)

include_directories(common)

add_subdirectory(ptx_inject)
add_subdirectory(stack_ptx)
add_subdirectory(stack_ptx_inject)
add_subdirectory(stack_ptx_inject_serialize)
