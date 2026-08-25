#include "tensor/Tensor.h"

Tensor Tensor::matmul(const Tensor& other) const {
    size_t size = shape_.size();
    size_t otherSize = other.shape_.size();
    if (shape_[size - 1] != other.shape_[otherSize - 2]) {
        throw std::runtime_error("The number of columns and rows must match for multiplication <matmul>");
    }

    std::vector<float> res(shape_[size - 2] * other.shape_[size - 1]);
    float sum = 0.0f;
    size_t index = 0;
    for (size_t row = 0; row < shape_[size - 2]; ++row) {
        for (size_t col = 0; col < other.shape_[size - 1]; ++col) {
            sum = 0.0f;
            for (size_t i = 0; i < shape_[size - 1]; ++ i) {
                sum += data_[row * strides_[size - 2] + i * strides_[size - 1]] * other.data_[i * other.strides_[size - 2] + col * other.strides_[size -1]];  
            }
            res[index++] = sum;
        }
    }

    return Tensor({shape_[size - 2], other.shape_[size - 1]}, res);
}