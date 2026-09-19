add_executable(test_tensor
    TensorArithmeticsTest.cpp
    TensorIndexingTest.cpp
    TensorLinearTest.cpp
    TensorShapeTest.cpp
    TensorTest.cpp
)

target_link_libraries(test_tensor
    PRIVATE
        tensor
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_tensor)