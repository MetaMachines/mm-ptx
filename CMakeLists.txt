cmake_minimum_required(VERSION 3.22)

project(mm-ptx)

find_package(CUDAToolkit REQUIRED)

if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA toolkit not found. Please ensure CUDA is installed.")
endif()

enable_language(CUDA)

string(REGEX MATCH "^[0-9]+" CUDA_MAJOR_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# cuContextCreate changed in cuda 13
if(CUDA_MAJOR_VERSION EQUAL 13) 
	add_definitions(-DCUDA_VERSION_13) 
endif()

add_library(mm_ptx_headers INTERFACE)

target_include_directories(mm_ptx_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}               # root (.)
    ${CMAKE_CURRENT_SOURCE_DIR}/generated_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

if(CUDA_MAJOR_VERSION EQUAL 13)
    target_compile_definitions(mm_ptx_headers INTERFACE CUDA_VERSION_13)
endif()

add_subdirectory(thirdparty)
add_subdirectory(tools)
add_subdirectory(tests)
add_subdirectory(examples)
