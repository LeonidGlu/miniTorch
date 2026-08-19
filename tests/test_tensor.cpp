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
    ::testing::ValuesIn(generateDataForBinaryOp(100, add_op, "addition"))
);

INSTANTIATE_TEST_SUITE_P(
    Subtraction,
    BinaryOpTest,
    ::testing::ValuesIn(generateDataForBinaryOp(100, sub_op, "subtraction"))
);

INSTANTIATE_TEST_SUITE_P(
    Multiplication,
    BinaryOpTest,
    ::testing::ValuesIn(generateDataForBinaryOp(100, mul_op, "multiplication"))
);

INSTANTIATE_TEST_SUITE_P(
    Division,
    BinaryOpTest,
    ::testing::ValuesIn(generateDataForBinaryOp(100, div_op, "division"))
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
    ::testing::ValuesIn(generateDataForCompoundOp(100, add_assign_op, "CompoundAdd"))
);

INSTANTIATE_TEST_SUITE_P(
    CompoundSub,
    CompoundOpTest,
    ::testing::ValuesIn(generateDataForCompoundOp(100, sub_assign_op, "CompoundSub"))
);

INSTANTIATE_TEST_SUITE_P(
    CompoundMul,
    CompoundOpTest,
    ::testing::ValuesIn(generateDataForCompoundOp(100, mul_assign_op, "CompoundMul"))
);

INSTANTIATE_TEST_SUITE_P(
    CompoundDiv,
    CompoundOpTest,
    ::testing::ValuesIn(generateDataForCompoundOp(100, div_assign_op, "CompoundDiv"))
);

TEST(TensorTest, Creation1){
    Tensor t({2,2});

    EXPECT_EQ(t.size(), 4);
    EXPECT_EQ(t.shape(), std::vector<size_t>({2,2}));
}

TEST(TensorTest, Creation2){
    Tensor t({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    EXPECT_FLOAT_EQ(t({0,0}), 1.0f);
    EXPECT_FLOAT_EQ(t({0,1}), 2.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 3.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 4.0f);
}

TEST(TensorTest, CopyConstructor){
    Tensor original({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor copy(original);
    EXPECT_FLOAT_EQ(copy({0,0}), 1.0f);
    EXPECT_FLOAT_EQ(copy({0,1}), 2.0f);
    EXPECT_FLOAT_EQ(copy({1,0}), 3.0f);
    EXPECT_FLOAT_EQ(copy({1,1}), 4.0f);

    copy({0, 0}) = 100.0f;
    EXPECT_FLOAT_EQ(original({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(copy({0, 0}), 100.0f);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.shape(), original.shape());
}

TEST(TensorTest, CopyAssignment){
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2, 2}, {12.0f, 22.0f, 32.0f, 42.0f});

    a = b;
    EXPECT_FLOAT_EQ(a({0,0}), 12.0f);
    EXPECT_FLOAT_EQ(a({0,1}), 22.0f);
    EXPECT_FLOAT_EQ(a({1,0}), 32.0f);
    EXPECT_FLOAT_EQ(a({1,1}), 42.0f);

    a({0, 0}) = 100.0f;
    EXPECT_FLOAT_EQ(b({0, 0}), 12.0f);
    EXPECT_FLOAT_EQ(a({0, 0}), 100.0f);

}

TEST(TensorTest, SelfCopyAssignment){
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    a = a;
    EXPECT_FLOAT_EQ(a({0,0}), 1.0f);
    EXPECT_FLOAT_EQ(a({0,1}), 2.0f);
    EXPECT_FLOAT_EQ(a({1,0}), 3.0f);
    EXPECT_FLOAT_EQ(a({1,1}), 4.0f);
}

TEST(TensorTest, MoveConstructor){
    Tensor original({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor moved(std::move(original));

    EXPECT_FLOAT_EQ(moved({0,0}), 1.0f);
    EXPECT_FLOAT_EQ(moved({0,1}), 2.0f);
    EXPECT_FLOAT_EQ(moved({1,0}), 3.0f);
    EXPECT_FLOAT_EQ(moved({1,1}), 4.0f);

    EXPECT_EQ(original.size(), 0);
    EXPECT_TRUE(original.shape().empty());
}

TEST(TensorTest, MoveAssignment){
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2, 2}, {12.0f, 22.0f, 32.0f, 42.0f});

    a = std::move(b);
    EXPECT_FLOAT_EQ(a({0,0}), 12.0f);
    EXPECT_FLOAT_EQ(a({0,1}), 22.0f);
    EXPECT_FLOAT_EQ(a({1,0}), 32.0f);
    EXPECT_FLOAT_EQ(a({1,1}), 42.0f);

    EXPECT_EQ(b.size(), 0);
    EXPECT_TRUE(b.shape().empty());
}

TEST(TensorTest, SelfMoveAssignment){
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    a = std::move(a);
    EXPECT_FLOAT_EQ(a({0,0}), 1.0f);
    EXPECT_FLOAT_EQ(a({0,1}), 2.0f);
    EXPECT_FLOAT_EQ(a({1,0}), 3.0f);
    EXPECT_FLOAT_EQ(a({1,1}), 4.0f);
}

TEST(TensorTest, Access){
    Tensor t({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    t({0,0}) = 10.0f;
    
    EXPECT_FLOAT_EQ(t({0,0}), 10.0f);
}

TEST(TensorTest, Matmul){
    Tensor a({2,3}, {1.0f, 4.0f, 12.0f, 10.0f, 2.0f, 6.0f});
    Tensor b({3,3}, {1.0f, 2.0f, 3.0f, 5.0f, 3.0f, 44.0f, 12.0f, 100.0f, 9.0f});
    Tensor t = a.matmul(b);

    std::vector<size_t> shapeA = a.shape();
    std::vector<size_t> shapeB = b.shape();
    std::vector<size_t> shapeT = t.shape();

    EXPECT_EQ(shapeT[0], shapeA[0]);
    EXPECT_EQ(shapeT[1], shapeB[1]);
    EXPECT_FLOAT_EQ(t({0,0}), 165.f);
    EXPECT_FLOAT_EQ(t({0,1}), 1214.0f);
    EXPECT_FLOAT_EQ(t({0,2}), 287.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 92.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 626.0f);
    EXPECT_FLOAT_EQ(t({1,2}), 172.0f);
}

