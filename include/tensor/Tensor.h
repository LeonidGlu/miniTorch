#include <iostream>
#include <vector>
#include <stdexcept>

class Tensor
{
public:
    Tensor(const std::vector<size_t> &shape);
    Tensor(const std::vector<size_t> &shape, const std::vector<float> data);
    Tensor(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    ~Tensor() = default;

    float &operator()(const std::vector<size_t> &indices);
    const float &operator()(const std::vector<size_t> &indices) const;
    Tensor& operator=(const Tensor &other);
    Tensor& operator=(Tensor &&other) noexcept;
    Tensor operator+(const Tensor &other) const;
    Tensor operator+(const float scalar) const;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator+=(const float scalar);
    Tensor operator-(const Tensor &other) const;
    Tensor operator-(const float scalar) const;
    Tensor& operator-=(const Tensor& other);
    Tensor& operator-=(const float scalar);
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(const float scalar) const;
    Tensor& operator*=(const Tensor& other);
    Tensor& operator*=(const float scalar);
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(const float scalar) const;
    Tensor& operator/=(const Tensor& other);
    Tensor& operator/=(const float scalar);

    Tensor matmul(const Tensor& other) const;

    void print() const;
    size_t size() const;
    const std::vector<size_t> &shape() const;

private:
    void initStrides();
    void printRecursive(size_t dim, std::vector<size_t> &indices, std::ostream &os, size_t indent) const;
    void printIndent(std::ostream &os, size_t indent) const;

    std::vector<float> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
};
