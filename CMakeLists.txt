cmake_minimum_required (VERSION 3.16)

project(miniTorch VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(src/tensor)

enable_testing()
find_package(GTest CONFIG REQUIRED)

add_executable(test_tensor
    tests/tensor_tests/TensorArithmeticsTest.cpp
    tests/tensor_tests/TensorIndexingTest.cpp
    tests/tensor_tests/TensorLinearTest.cpp
    tests/tensor_tests/TensorShapeTest.cpp
    tests/tensor_tests/TensorTest.cpp
)

target_link_libraries(test_tensor
    PRIVATE
        tensor
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_tensor)

