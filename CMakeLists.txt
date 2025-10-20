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

include_directories(generated_headers)
include_directories(helpers)
include_directories(thirdparty)
include_directories(.)

add_subdirectory(tools/ptxinject)
add_subdirectory(tests)
add_subdirectory(examples)
