# cmake/mm_ptx_codegen.cmake

function(mm_add_stack_ptx_header)
  set(options)
  set(oneValueArgs OUT_HDR JSON SCRIPT LANG)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(MM "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT MM_OUT_HDR)
    message(FATAL_ERROR "mm_add_stack_ptx_header: OUT_HDR is required")
  endif()
  if(NOT MM_JSON)
    message(FATAL_ERROR "mm_add_stack_ptx_header: JSON is required")
  endif()

  if(NOT MM_SCRIPT)
    set(MM_SCRIPT "${MM_PTX_TOOLS_DIR}/stack_ptx_generate_infos.py")
  endif()

  if(NOT MM_LANG)
    set(MM_LANG c)
  endif()

  get_filename_component(_out_dir "${MM_OUT_HDR}" DIRECTORY)

  add_custom_command(
    OUTPUT  "${MM_OUT_HDR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_out_dir}"
    COMMAND ${Python3_EXECUTABLE} "${MM_SCRIPT}"
            --input  "${MM_JSON}"
            --output "${MM_OUT_HDR}"
            --lang   "${MM_LANG}"
    DEPENDS "${MM_JSON}" "${MM_SCRIPT}" ${MM_DEPENDS}
    COMMENT "Generating Stack-PTX header from ${MM_JSON}"
    VERBATIM
  )

  set_source_files_properties("${MM_OUT_HDR}"
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
  )

  set(${MM_OUT_HDR} "${MM_OUT_HDR}" PARENT_SCOPE)
endfunction()
