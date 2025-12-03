set(MM_PTX_CODEGEN_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Template used to build ptxinject type-registry plugins
set(MM_PTX_TYPES_PLUGIN_TEMPLATE
    "${MM_PTX_CODEGEN_DIR}/mm_ptx_custom_type_plugin.c.in"
)

# mm_add_ptxinject_types_plugin:
#
# Build a MODULE library that exposes a PtxInjectTypeRegistry based on a
# generated types header.
#
# Required:
#   TARGET       : name of the plugin target to create (e.g. "custom_types")
#   TYPES_HEADER : full path to generated types header (.h)
#
# Optional:
#   GENERATED_INC_DIR  : directory where TYPES_HEADER lives
#                        (inferred from TYPES_HEADER if omitted)
#   EXTRA_INCLUDE_DIRS : extra include dirs
#   LINK_LIBS          : extra libs (mm_ptx_headers is always linked)
#
function(mm_add_ptxinject_types_plugin)
  set(options)
  set(oneValueArgs
      TARGET
      TYPES_HEADER
  )
  set(multiValueArgs
      LINK_LIBS
  )
  cmake_parse_arguments(MM "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT MM_TARGET)
    message(FATAL_ERROR "mm_add_ptxinject_types_plugin: TARGET is required")
  endif()
  if(NOT MM_TYPES_HEADER)
    message(FATAL_ERROR "mm_add_ptxinject_types_plugin: TYPES_HEADER is required")
  endif()

  # Directory where the generated header lives
  get_filename_component(_generated_inc_dir "${MM_TYPES_HEADER}" DIRECTORY)

  # Sanity-check the template exists
  if(NOT EXISTS "${MM_PTX_TYPES_PLUGIN_TEMPLATE}")
    message(FATAL_ERROR
      "mm_add_ptxinject_types_plugin: Template '${MM_PTX_TYPES_PLUGIN_TEMPLATE}' "
      "not found. Expected it next to mm_ptx_codegen.cmake."
    )
  endif()

  # Where to write the configured plugin source
  set(_plugin_src "${CMAKE_CURRENT_BINARY_DIR}/${MM_TARGET}_plugin.c")

  # Pass just the basename of the header into the template
  get_filename_component(_types_header_basename "${MM_TYPES_HEADER}" NAME)
  set(PTX_TYPES_HEADER_BASENAME "${_types_header_basename}")

  configure_file("${MM_PTX_TYPES_PLUGIN_TEMPLATE}" "${_plugin_src}" @ONLY)

  # MODULE library implementing the registry
  add_library(${MM_TARGET} MODULE "${_plugin_src}")

  # Includes:
  #  - generated header dir (so #include "<basename>" works)
  #  - cmake/ dir (for any shared headers next to the template)
  #  - tools dir for ptxinject headers
  target_include_directories(${MM_TARGET}
    PRIVATE
      "${_generated_inc_dir}"
      "${MM_PTX_CODEGEN_DIR}"
      "${MM_PTX_TOOLS_DIR}/ptxinject"
  )

  target_link_libraries(${MM_TARGET}
    PRIVATE
      mm_ptx_headers
      ${MM_LINK_LIBS}
  )

  # Rebuild plugin when the generated header changes
  set_source_files_properties("${_plugin_src}"
    PROPERTIES OBJECT_DEPENDS "${MM_TYPES_HEADER}"
  )
endfunction()

