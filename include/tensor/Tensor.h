#include <iostream>
#include <vector>
#include <stdexcept>

class Tensor
{
public:
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const std::vector<float> data);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor() = default;

    float& operator()(size_t i, size_t j);
    const float& operator()(size_t i, size_t j) const;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor operator+(const Tensor& other) const;
    Tensor operator+(const float otherScalar) const;

    void print() const;
    size_t size() const;
    const std::vector<size_t>& shape() const;

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    size_t size_;

};
