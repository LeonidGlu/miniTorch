#include "tensor/Tensor.h"

float& Tensor::operator()(const std::vector<size_t>& indices) {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("The number of indexes must match the dimension <operator()>");
    }
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds <operator()>");
        }
        index += indices[i] * strides_[i];
    }

    return data_[index];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("The number of indexes must match the dimension <operator()>");
    }
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds <operator()>");
        }
        index += indices[i] * strides_[i];
    }

    return data_[index];
}