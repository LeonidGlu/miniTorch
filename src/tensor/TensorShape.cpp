#include "tensor/Tensor.h"

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