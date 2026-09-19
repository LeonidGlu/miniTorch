#include <gtest/gtest.h>
#include "tensor/Tensor.h"
#include <random>

struct BinaryOpTestParams{
    std::string name;
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::vector<float> expected;
    Tensor (*op)(const Tensor&, const Tensor&);
};

Tensor add_op(const Tensor& x, const Tensor& y) {
    std::vector<float> dataX = x.data();
    std::vector<float> dataY = y.data();
    std::vector<float> result(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = dataX[i] + dataY[i];
    }
    return Tensor(x.shape(), result); 
}

Tensor sub_op(const Tensor& x, const Tensor& y) {
    std::vector<float> dataX = x.data();
    std::vector<float> dataY = y.data();
    std::vector<float> result(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = dataX[i] - dataY[i];
    }
    return Tensor(x.shape(), result); 
}

Tensor mul_op(const Tensor& x, const Tensor& y) {
    std::vector<float> dataX = x.data();
    std::vector<float> dataY = y.data();
    std::vector<float> result(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = dataX[i] * dataY[i];
    }
    return Tensor(x.shape(), result); 
}

Tensor div_op(const Tensor& x, const Tensor& y) {
    std::vector<float> dataX = x.data();
    std::vector<float> dataY = y.data();
    std::vector<float> result(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = dataX[i] / dataY[i];
    }
    return Tensor(x.shape(), result); 
}

Tensor random_tensor(const std::vector<size_t>& shape) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    size_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    } 
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    return Tensor(shape, data);
}

std::vector<BinaryOpTestParams> generateDataForBinaryOp(size_t count, Tensor (*op)(const Tensor&, const Tensor&), const std::string& opName) {
    std::vector<BinaryOpTestParams> test_cases;

    for (size_t i = 0; i < count; ++i) {
        auto a = random_tensor({2, 2});
        auto b = random_tensor({2, 2});

        Tensor result = op(a, b);
        std::vector<float> expected(result.size());
        for (size_t j = 0; j < result.size(); ++j) {
            expected[j] = result.data()[j];
        }

        test_cases.push_back({opName + "_" + std::to_string(i), std::make_shared<Tensor>(a), std::make_shared<Tensor>(b), expected, op});
    }
    return test_cases;
}

class BinaryOpTest : public ::testing::TestWithParam<BinaryOpTestParams> {
public:
    BinaryOpTest() = default;

protected:
    void SetUp() override {
        const auto& params = GetParam();
        a = params.a;
        b = params.b;
        expected = params.expected;
        op = params.op;
    }

    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::vector<float> expected;
    Tensor (*op)(const Tensor&, const Tensor&);
};

TEST_P(BinaryOpTest, BinaryOperations){
    Tensor result = op(*a, *b);
    for (size_t i = 0; i < a->size(); ++i) {
        EXPECT_FLOAT_EQ(result.data()[i], expected[i]) << "Failed for: " << GetParam().name;
    }
       
}

INSTANTIATE_TEST_SUITE_P(
    Addition,
    BinaryOpTest,
    ::testing::ValuesIn(generateDataForBinaryOp(10, add_op, "addition"))
);

INSTANTIATE_TEST_SUITE_P(
    Subtraction,
    BinaryOpTest,
    ::testing::ValuesIn(generateDataForBinaryOp(10, sub_op, "subtraction"))
);

INSTANTIATE_TEST_SUITE_P(
    Multiplication,
    BinaryOpTest,
    ::testing::ValuesIn(generateDataForBinaryOp(10, mul_op, "multiplication"))
);

INSTANTIATE_TEST_SUITE_P(
    Division,
    BinaryOpTest,
    ::testing::ValuesIn(generateDataForBinaryOp(10, div_op, "division"))
);

struct CompoundOpTestParams{
    std::string name;
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::vector<float> expected;
    void (*op)(Tensor&, const Tensor&);
};

void add_assign_op(Tensor& x, const Tensor& y) { x += y; };
void sub_assign_op(Tensor& x, const Tensor& y) { x -= y; };
void mul_assign_op(Tensor& x, const Tensor& y) { x *= y; };
void div_assign_op(Tensor& x, const Tensor& y) { x /= y; };

std::vector<CompoundOpTestParams> generateDataForCompoundOp(size_t count, void (*op)(Tensor&, const Tensor&), const std::string& opName) {
    std::vector<CompoundOpTestParams> test_cases;

    for (size_t i = 0; i < count; ++i) {
        auto a = random_tensor({2, 2});
        auto b = random_tensor({2, 2});
        Tensor copyA = a;
        op(a, b);
        std::vector<float> expected(a.size());
        for (size_t j = 0; j < a.size(); ++j) {
            expected[j] = a.data()[j];
        }

        test_cases.push_back({opName + "_" + std::to_string(i), std::make_shared<Tensor>(copyA), std::make_shared<Tensor>(b), expected, op});
    }

    return test_cases;
}

class CompoundOpTest : public ::testing::TestWithParam<CompoundOpTestParams> {
public:
    CompoundOpTest() = default;

protected:
    void SetUp() override {
        const auto& params = GetParam();
        a = params.a;
        b = params.b;
        expected = params.expected;
        op = params.op;
    }

    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::vector<float> expected;
    void (*op)(Tensor&, const Tensor&);
};

TEST_P(CompoundOpTest, CompoundOperations) {
    op(*a, *b);
    Tensor result = *a;
    for (size_t i = 0; i < a->size(); ++i) {
        EXPECT_FLOAT_EQ(result.data()[i], expected[i]) << "Failed for: " << GetParam().name;
    }
}

INSTANTIATE_TEST_SUITE_P(
    CompoundAdd,
    CompoundOpTest,
    ::testing::ValuesIn(generateDataForCompoundOp(10, add_assign_op, "CompoundAdd"))
);

INSTANTIATE_TEST_SUITE_P(
    CompoundSub,
    CompoundOpTest,
    ::testing::ValuesIn(generateDataForCompoundOp(10, sub_assign_op, "CompoundSub"))
);

INSTANTIATE_TEST_SUITE_P(
    CompoundMul,
    CompoundOpTest,
    ::testing::ValuesIn(generateDataForCompoundOp(10, mul_assign_op, "CompoundMul"))
);

INSTANTIATE_TEST_SUITE_P(
    CompoundDiv,
    CompoundOpTest,
    ::testing::ValuesIn(generateDataForCompoundOp(10, div_assign_op, "CompoundDiv"))
);