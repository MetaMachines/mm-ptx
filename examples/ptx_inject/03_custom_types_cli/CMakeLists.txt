project(ptx_inject_03_custom_types_cli)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(GENERATED_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(PTX_TYPES_JSON   "${CMAKE_CURRENT_SOURCE_DIR}/custom_types.json")
set(GENERATED_HDR     "${GENERATED_INC_DIR}/custom_types.h")

mm_add_ptx_inject_header(
  OUT_HDR "${GENERATED_HDR}"
  JSON    "${PTX_TYPES_JSON}"
)

mm_add_ptxinject_types_plugin(
  TARGET          custom_types
  TYPES_HEADER    "${GENERATED_HDR}"
)

create_injected_ptx(
  ANNOTATED_PTX
  kernel.cu
  PLUGIN_TARGET custom_types
)

add_executable(${PROJECT_NAME} main.c)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.c OBJECT_DEPENDS ${ANNOTATED_PTX})
target_compile_definitions(${PROJECT_NAME} PRIVATE "PTX_KERNEL=${ANNOTATED_PTX}")
target_link_libraries(
    ${PROJECT_NAME}
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
)

target_include_directories(${PROJECT_NAME} PRIVATE "${GENERATED_INC_DIR}")
