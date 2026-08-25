#include "tensor/Tensor.h"

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

Tensor Tensor::operator+(const float scalar) const {
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = data_[i] + scalar;
    }
    return Tensor(shape_, res);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (other.shape_ != shape_) {
        throw std::runtime_error("The dimensions must be match <operator->");
    }
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = data_[i] - other.data_[i];
    }
    return Tensor(shape_, res);
}

Tensor Tensor::operator-(const float scalar) const {
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = data_[i] - scalar;
    }
    return Tensor(shape_, res);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (other.shape_ != shape_) {
        throw std::runtime_error("The dimensions must be match <operator*>");
    }
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = data_[i] * other.data_[i];
    }
    return Tensor(shape_, res);
}

Tensor Tensor::operator*(const float scalar) const {
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = data_[i] * scalar;
    }
    return Tensor(shape_, res);
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (other.shape_ != shape_) {
        throw std::runtime_error("The dimensions must be match <operator/>");
    }
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        if (other.data_[i] == 0) {
            throw std::runtime_error("It cannot be divided by 0 <operator/>");
        }
        res[i] = data_[i] / other.data_[i];
    }
    return Tensor(shape_, res);
}

Tensor Tensor::operator/(const float scalar) const {
    if (scalar == 0) {
        throw std::runtime_error("It cannot be divided by 0 <operator/>");
    }
    std::vector<float> res(size_);
    for (size_t i = 0; i < size_; ++i) {
        res[i] = data_[i] / scalar;
    }
    return Tensor(shape_, res);
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (other.shape_ != shape_) {
        throw std::runtime_error("The dimensions must be match <operator+=>");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator+=(const float scalar) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] += scalar;
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (other.shape_ != shape_) {
        throw std::runtime_error("The dimensions must be match <operator-=>");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const float scalar) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] -= scalar;
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (other.shape_ != shape_) {
        throw std::runtime_error("The dimensions must be match <operator*=>");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] *=  other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const float scalar) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (other.shape_ != shape_) {
        throw std::runtime_error("The dimensions must be match <operator/=>");
    }
    for (size_t i = 0; i < size_; ++i) {
        if (other.data_[i] == 0) {
            throw std::runtime_error("It cannot be divided by 0 <operator/=>");
        }
        data_[i] /= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const float scalar) {
    if (scalar == 0) {
        throw std::runtime_error("It cannot be divided by 0 <operator/=>");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] /= scalar;
    }
    return *this;
}