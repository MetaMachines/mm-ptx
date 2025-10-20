cmake_minimum_required(VERSION 3.10)
project(ptxinject C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(.)

# Add executable
add_executable(${PROJECT_NAME} main.c)

# Platform-specific (e.g., for Windows polyfills)
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _WIN32)
    # Add any Windows-specific sources if needed
endif()

# Install
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
