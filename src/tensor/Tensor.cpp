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


void Tensor::print() const {
    std::vector<size_t> indices(shape_.size(), 0);
    printRecursive(0, indices, std::cout, 0);
    std::cout << "\n"; 
}

void Tensor::printRecursive(size_t dim, std::vector<size_t>& indices, std::ostream& os, size_t indent) const {
    if (dim == shape_.size()) {
        os << (*this)(indices);
        return;
    }

    os << "[";
    bool is_last_dim = (dim == shape_.size() - 1);
    if (!is_last_dim) {
        os << "\n";
        printIndent(os, indent + 2);
    }

    for(size_t i = 0; i < shape_[dim]; ++i) {
        indices[dim] = i;
        printRecursive(dim + 1, indices, os, indent + 2);

        if (i != shape_[dim] - 1) {
            os << ", ";
            if (!is_last_dim) {
                os << "\n";
                printIndent(os, indent + 2);
            }
        }
    }

    if (!is_last_dim) {
        os << "\n";
        printIndent(os, indent);
    }
    os << "]";
}

void Tensor::printIndent(std::ostream& os, size_t indent) const {
    for (size_t i = 0; i < indent; ++i) {
        os << " ";
    }
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