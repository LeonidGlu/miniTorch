#include "tensor/Tensor.h"

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    if (shape_.empty()) {
        data_.resize(1,0);
        size_ = 1;
    } else {
        size_ = 1;
        for (auto i : shape_) {
            size_ *= i;
        }
        data_.resize(size_, 0);
    }
    initStrides();
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float> data) : shape_(shape), data_(data) {
    if (shape.empty()) {
        size_ = 1;
    } else {
        size_ = 1;
        for (auto i : shape_) {
            size_ *= i;
        }
    }
    if (data_.size() != size_) {
        throw std::runtime_error("Data size doesn't match shape");
    }
    initStrides();
}

Tensor::Tensor(const Tensor& other) {
    data_ = other.data_;
    shape_ = other.shape_;
    size_ = other.size_;
    strides_ = other.strides_;
}

Tensor::Tensor(Tensor&& other) noexcept : data_(std::move(other.data_)),  shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), size_(other.size_) {
    other.size_ = 0;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        size_ = other.size_;
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        size_ = other.size_;
        other.size_ = 0;
    }

    return *this;
}

size_t Tensor::size() const {
    return size_;
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<float>& Tensor::data() const {
    return data_;
}