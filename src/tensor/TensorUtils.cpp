#include "tensor/Tensor.h"

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