#include <gtest/gtest.h>
#include "tensor/Tensor.h"

struct TransposeTestParams {
    std::string name;
    std::vector<size_t> shape;
    std::vector<float> data;
    std::vector<size_t> expected_shape;
    std::vector<float> expected_data;
};

class TransposeTest : public ::testing::TestWithParam<TransposeTestParams> {
public:
    TransposeTest() = default;

protected:
    void SetUp() override {
        const auto& params = GetParam();
        shape = params.shape;
        data = params.data;
        expected_shape = params.expected_shape;
        expected_data = params.expected_data;
    }

    std::vector<size_t> shape;
    std::vector<float> data;
    std::vector<size_t> expected_shape;
    std::vector<float> expected_data;
};

TEST_P(TransposeTest, Transpose) {
    const auto& params = GetParam();
    Tensor a(params.shape, params.data);

    a.transpose();
    EXPECT_EQ(a.data(), params.data);
    EXPECT_EQ(a.shape(), params.expected_shape);

    size_t total = params.expected_data.size();
    std::vector<size_t> indices(params.expected_shape.size(), 0);
    size_t index = 0;

    for (size_t flat = 0; flat < total; ++flat) {
        size_t remainder = flat;
        for (int d = params.expected_shape.size() - 1; d >= 0; --d) {
            indices[d] = remainder % params.expected_shape[d];
            remainder /= params.expected_shape[d];
        }
    
        EXPECT_FLOAT_EQ(a(indices), params.expected_data[flat]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Transpose2D,
    TransposeTest,
    ::testing::Values(
        TransposeTestParams{"T2D", {2, 2}, {1, 2, 3, 4}, {2, 2}, {1, 3, 2, 4}},
        TransposeTestParams{"T2x3D", {2, 3}, {1, 2, 3, 4, 5, 6}, {3, 2}, {1, 4, 2, 5, 3, 6}},
        TransposeTestParams{"T3D", {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3}, {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12}}
    )
);

struct TransposeAxesTestParams {
    std::string name;
    std::vector<size_t> axes;
    std::vector<size_t> shape;
    std::vector<float> data;
    std::vector<size_t> expected_shape;
    std::vector<float> expected_data;
};

class TransposeAxesTest : public ::testing::TestWithParam<TransposeAxesTestParams> {
public:
    TransposeAxesTest() = default;

protected:
    void SetUp() override {
        const auto& params = GetParam();
        axes = params.axes;
        shape = params.shape;
        data = params.data;
        expected_shape = params.expected_shape;
        expected_data = params.expected_data;
    }

    std::vector<size_t> axes;
    std::vector<size_t> shape;
    std::vector<float> data;
    std::vector<size_t> expected_shape;
    std::vector<float> expected_data;
};

TEST_P(TransposeAxesTest, TransposeAxes) {
    const auto& params = GetParam();
    Tensor a(params.shape, params.data);

    a.transpose(params.axes);
    EXPECT_EQ(a.data(), params.data);
    EXPECT_EQ(a.shape(), params.expected_shape);

    size_t total = params.expected_data.size();
    std::vector<size_t> indices(params.expected_shape.size(), 0);
    size_t index = 0;

    for (size_t flat = 0; flat < total; ++flat) {
        size_t remainder = flat;
        for (int d = params.expected_shape.size() - 1; d >= 0; --d) {
            indices[d] = remainder % params.expected_shape[d];
            remainder /= params.expected_shape[d];
        }
    
        EXPECT_FLOAT_EQ(a(indices), params.expected_data[flat]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TransposeAxes,
    TransposeAxesTest,
    ::testing::Values(
        TransposeAxesTestParams{"T3DAxes", {1, 0, 2}, {3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2}, {1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12}},
        TransposeAxesTestParams{"T3DAxes2", {2, 1, 0}, {3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3}, {1, 5, 9, 3, 7, 11, 2, 6, 10, 4, 8, 12}}
    )
);