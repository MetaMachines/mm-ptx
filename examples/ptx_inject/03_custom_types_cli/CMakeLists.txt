project(ptx_inject_03_custom_types_cli)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PTX_TYPES_JSON   "${CMAKE_CURRENT_SOURCE_DIR}/custom_types.json" CACHE FILEPATH "Path to PTX Inject data_type_infos JSON")
set(PTX_GEN_SCRIPT   "${MM_PTX_TOOLS_DIR}/ptx_inject_generate_infos.py" CACHE FILEPATH "Path to header generator script")

set(GENERATED_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GENERATED_HDR     "${GENERATED_INC_DIR}/custom_types.h")

# Generate header file from JSON.
add_custom_command(
  OUTPUT  ${GENERATED_HDR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_INC_DIR}
  COMMAND ${Python3_EXECUTABLE} ${PTX_GEN_SCRIPT}
          --in ${PTX_TYPES_JSON}
          --out ${GENERATED_HDR}
  DEPENDS ${PTX_TYPES_JSON} ${PTX_GEN_SCRIPT}
  COMMENT "Generating PTX inject headers from ${PTX_TYPES_JSON}"
  VERBATIM
)

set_source_files_properties(${GENERATED_HDR}
  PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)

add_custom_target(ptx_types_headers DEPENDS ${GENERATED_HDR})

include_directories(.)
include_directories(${MM_PTX_TOOLS_DIR}/ptxinject)
include_directories("${GENERATED_INC_DIR}")

# Compile plugin using the registry.
add_library(custom_types MODULE custom_types_plugin.c) # .so/.dylib/.dll

target_link_libraries(
  custom_types
  mm_ptx_headers
)
target_include_directories(custom_types PRIVATE .)
target_include_directories(custom_types PRIVATE "${GENERATED_INC_DIR}")
add_dependencies(custom_types ptx_types_headers)

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_types_plugin.c
    PROPERTIES
    OBJECT_DEPENDS ${GENERATED_HDR}
)

set(PROCESSED_CUDA ${CMAKE_CURRENT_BINARY_DIR}/processed_cuda/kernel.cu)
set(ANNOTATED_PTX ${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx/kernel.ptx)
make_directory(${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx)

# Send the compiled registry to the `ptxinject` cli tool so it has your custom types.
add_custom_command(OUTPUT ${PROCESSED_CUDA}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu ptxinject
    COMMAND 
        $<TARGET_FILE:ptxinject> 
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu 
        -o ${CMAKE_CURRENT_BINARY_DIR}/processed_cuda
        -t $<TARGET_FILE:custom_types>
        -f
)

# Compile the processed CUDA to PTX.
add_custom_command(OUTPUT ${ANNOTATED_PTX}
    DEPENDS ${PROCESSED_CUDA}
    COMMAND nvcc -O3 -prec-sqrt=false -arch=native -ptx ${PROCESSED_CUDA} -o ${ANNOTATED_PTX}
)

# Compile the executable.
add_executable(${PROJECT_NAME} main.c)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/main.c OBJECT_DEPENDS ${ANNOTATED_PTX})
# Make the environment variable hold the path to the annotated PTX for INCBIN.
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
