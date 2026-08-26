#include "tensor/Tensor.h"
#include <set>
#include <algorithm>

void Tensor::initStrides() {
    if(shape_.empty()) {
        return;
    }

    strides_.resize(shape_.size());
    strides_.back() = 1;
    for(int i = (int)shape_.size() - 2; i >= 0; --i) {
        strides_[i] = shape_[i + 1] * strides_[i + 1];
    }
}

void Tensor::transpose() {
    if (shape_.size() < 2) {
        throw std::runtime_error("It is possible to transpose tensors that have at least 2 dimensions!");
    }
    std::swap(strides_[strides_.size() - 1], strides_[strides_.size() - 2]);
    std::swap(shape_[shape_.size()-1], shape_[shape_.size() - 2]);
}

void Tensor::transpose(std::initializer_list<size_t> axes) {
    if (shape_.size() < 2) {
        throw std::runtime_error("It is possible to transpose tensors that have at least 2 dimensions!");
    }
    if (axes.size() != shape_.size()) {
        throw std::runtime_error("The number of axes during transposition must match the dimensionality!");
    }

    std::set<size_t> unique_axes(axes.begin(), axes.end());
    if (unique_axes.size() != axes.size()) {
        throw std::runtime_error("Axes must be unique");
    }
    for (size_t axis : axes) {
        if (axis >= shape_.size()) {
            throw std::runtime_error("Axis out of bounds");
        }
    }

    std::vector<size_t> newShape;
    std::vector<size_t> newStrides;
    newShape.reserve(shape_.size());
    newStrides.reserve(strides_.size());

    for (size_t axis : axes) {
        newStrides.push_back(strides_[axis]);
        newShape.push_back(shape_[axis]);
    }
    shape_ = std::move(newShape);
    strides_ = std::move(newStrides);
}