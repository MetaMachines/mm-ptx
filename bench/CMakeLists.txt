set(MM_PTX_BENCH_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(STACK_PTX_DESCRIPTIONS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/stack_ptx_descriptions.json")
set(STACK_PTX_GENERATED_C_HDR "${MM_PTX_BENCH_GENERATED_DIR}/stack_ptx_example_descriptions.h")

mm_add_stack_ptx_header(
    OUT_HDR "${STACK_PTX_GENERATED_C_HDR}"
    JSON    "${STACK_PTX_DESCRIPTIONS_JSON}"
    LANG    c
)

add_custom_target(stack_ptx_bench_generated_headers
  DEPENDS
    "${STACK_PTX_GENERATED_C_HDR}"
)

add_library(stack_ptx_bench_generated INTERFACE)
target_include_directories(stack_ptx_bench_generated INTERFACE "${MM_PTX_BENCH_GENERATED_DIR}")
add_dependencies(stack_ptx_bench_generated stack_ptx_bench_generated_headers)

if(TARGET CUDA::nvptxcompiler_static)
  set(MM_PTX_BENCH_NVPTX_TARGET CUDA::nvptxcompiler_static)
elseif(TARGET CUDA::nvptxcompiler)
  set(MM_PTX_BENCH_NVPTX_TARGET CUDA::nvptxcompiler)
else()
  message(FATAL_ERROR "CUDAToolkit nvptxcompiler library not found.")
endif()

add_executable(mm_ptx_compile_bench
  ptx_compile_bench.c
)

target_include_directories(mm_ptx_compile_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../common"
)

target_link_libraries(mm_ptx_compile_bench PRIVATE
  CUDA::cuda_driver
  CUDA::nvrtc
  ${MM_PTX_BENCH_NVPTX_TARGET}
  m
  mm_ptx_headers
  stack_ptx_bench_generated
)

target_compile_features(mm_ptx_compile_bench PRIVATE c_std_11)
