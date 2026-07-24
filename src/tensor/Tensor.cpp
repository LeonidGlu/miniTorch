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
}

Tensor::Tensor(const Tensor& other) {
    data_ = other.data_;
    shape_ = other.shape_;
    size_ = other.size_;
}

Tensor::Tensor(Tensor&& other) noexcept : data_(std::move(other.data_)),  shape_(std::move(other.shape_)), size_(other.size_) {
    other.size_ = 0;
}

float& Tensor::operator()(size_t i, size_t j) {
    if (i >= shape_[0] || j >= shape_[1]) {
        throw std::runtime_error("Index out of bounds <operator()>");
    }
    return data_[i * shape_[1] + j];
}

const float& Tensor::operator()(size_t i, size_t j) const {
    if (i >= shape_[0] || j >= shape_[1]) {
        throw std::runtime_error("Index out of bounds <operator()>");
    }
    return data_[i * shape_[1] + j];
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
        size_ = other.size_;
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        other.size_ = 0;
    }

    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (other.shape_ != shape_) {
        throw std::runtime_error("The dimensions must be match <operator+>");
    }
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = data_[i] + other.data_[i];
    }
    return Tensor(shape_, res);
}

Tensor Tensor::operator+(const float otherScalar) const {
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = data_[i] + otherScalar;
    }
    return Tensor(shape_, res);
}

void Tensor::print() const {
    if (shape_.empty()) {
        std::cout << "[ " << data_[0] << " ]\n";
    }
    else {
        for (size_t row = 0; row < shape_[0]; ++row) {
            std::cout << "[ ";
            for (size_t col = 0; col < shape_[1]; ++col) {
                std::cout << (*this)(row, col) << " ";
            }
            std::cout << " ]\n";
        }
    }   
}

size_t Tensor::size() const {
    return size_;
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}