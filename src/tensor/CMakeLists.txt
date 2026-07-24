add_library(tensor 
    Tensor.cpp
)

target_include_directories(tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)