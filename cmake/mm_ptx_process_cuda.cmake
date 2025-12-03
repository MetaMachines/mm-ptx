#
# Creates custom commands to process a .cu file with ptxinject and then
# compile to .ptx with nvcc.
#
# Usage:
#   create_injected_ptx(OUT_VAR kernel.cu)
#   create_injected_ptx(OUT_VAR kernel.cu PLUGIN_TARGET custom_types)
#   create_injected_ptx(OUT_VAR kernel.cu
#                       PLUGIN_TARGET custom_types
#                       NVCC_INCLUDE_DIRS ${MM_PTX_CUTLASS_INCLUDE_DIR} /some/other/include)
#
# Positional:
#   OUT_VAR      - variable name that will receive the final .ptx path
#   CUDA_SOURCE  - path to the input .cu file (relative or absolute)
#
# Optional named args:
#   PLUGIN_TARGET     - name of a CMake target to pass via -t $<TARGET_FILE:...>
#   NVCC_INCLUDE_DIRS - one or more include dirs to pass to nvcc as -I <dir>
#
function(create_injected_ptx OUTPUT_VAR CUDA_SOURCE)
    set(options)
    set(oneValueArgs PLUGIN_TARGET)
    set(multiValueArgs NVCC_INCLUDE_DIRS)
    cmake_parse_arguments(CIP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Get the full path to the source file
    if (IS_ABSOLUTE "${CUDA_SOURCE}")
        set(FULL_CUDA_SOURCE "${CUDA_SOURCE}")
    else()
        set(FULL_CUDA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}")
    endif()

    # Base name: kernel.cu -> kernel
    get_filename_component(SOURCE_BASENAME "${CUDA_SOURCE}" NAME_WE)

    # Intermediate / output paths
    set(PROCESSED_DIR   "${CMAKE_CURRENT_BINARY_DIR}/processed_cuda")
    set(PTX_DIR         "${CMAKE_CURRENT_BINARY_DIR}/annotated_ptx")
    set(PROCESSED_CUDA  "${PROCESSED_DIR}/${SOURCE_BASENAME}.cu")
    set(ANNOTATED_PTX   "${PTX_DIR}/${SOURCE_BASENAME}.ptx")

    # Ensure dirs exist at configure time
    file(MAKE_DIRECTORY "${PROCESSED_DIR}")
    file(MAKE_DIRECTORY "${PTX_DIR}")

    # Build nvcc include flags from NVCC_INCLUDE_DIRS
    set(_nvcc_includes "")
    foreach(inc IN LISTS CIP_NVCC_INCLUDE_DIRS)
        list(APPEND _nvcc_includes "-I" "${inc}")
    endforeach()

    # -------- ptxinject step --------
    if (CIP_PLUGIN_TARGET)
        add_custom_command(
            OUTPUT  "${PROCESSED_CUDA}"
            DEPENDS "${FULL_CUDA_SOURCE}" ptxinject ${CIP_PLUGIN_TARGET}
            COMMAND $<TARGET_FILE:ptxinject>
                    "${FULL_CUDA_SOURCE}"
                    -o "${PROCESSED_DIR}"
                    -t $<TARGET_FILE:${CIP_PLUGIN_TARGET}>
                    -f
            COMMENT "Injecting PTX for ${CUDA_SOURCE} with plugin ${CIP_PLUGIN_TARGET}"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT  "${PROCESSED_CUDA}"
            DEPENDS "${FULL_CUDA_SOURCE}" ptxinject
            COMMAND $<TARGET_FILE:ptxinject>
                    "${FULL_CUDA_SOURCE}"
                    -o "${PROCESSED_DIR}"
                    -f
            COMMENT "Injecting PTX for ${CUDA_SOURCE}"
            VERBATIM
        )
    endif()

    # -------- nvcc step --------
    add_custom_command(
        OUTPUT  "${ANNOTATED_PTX}"
        DEPENDS "${PROCESSED_CUDA}"
        COMMAND nvcc -O3 -prec-sqrt=false -arch=native -ptx
                "${PROCESSED_CUDA}" -o "${ANNOTATED_PTX}"
                ${_nvcc_includes}
        COMMENT "Compiling processed ${SOURCE_BASENAME}.cu to PTX"
        VERBATIM
    )

    # Return the PTX path to the caller
    set(${OUTPUT_VAR} "${ANNOTATED_PTX}" PARENT_SCOPE)
endfunction()
