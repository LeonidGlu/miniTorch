add_library(tensor 
    Tensor.cpp
    TensorArithmetics.cpp
    TensorIndexing.cpp
    TensorShape.cpp
    TensorUtils.cpp
    TensorLinearAlgebra.cpp
)

target_include_directories(tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)