cmake_minimum_required (VERSION 3.16)

project(miniTorch VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)

add_subdirectory(src/tensor)

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
endif()