#include <gtest/gtest.h>
#include "tensor/Tensor.h"

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

TEST(TensorTest, Access){
    Tensor t({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    t({0,0}) = 10.0f;
    
    EXPECT_FLOAT_EQ(t({0,0}), 10.0f);
}

TEST(TensorTest, Addition){
    Tensor a({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor t = a + b;

    EXPECT_FLOAT_EQ(t({0,0}), 2.0f);
    EXPECT_FLOAT_EQ(t({0,1}), 4.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 6.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 8.0f);
}

TEST(TensorTest, AdditionWithScalar){
    Tensor a({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    float scalar = 5.0f;
    Tensor t = a + scalar;

    EXPECT_FLOAT_EQ(t({0,0}), 6.0f);
    EXPECT_FLOAT_EQ(t({0,1}), 7.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 8.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 9.0f);

}

TEST(TensorTest, Subtraction){
    Tensor a({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2,2}, {2.0f, 4.0f, 6.0f, 8.0f});
    Tensor t = a - b;

    EXPECT_FLOAT_EQ(t({0,0}), -1.0f);
    EXPECT_FLOAT_EQ(t({0,1}), -2.0f);
    EXPECT_FLOAT_EQ(t({1,0}), -3.0f);
    EXPECT_FLOAT_EQ(t({1,1}), -4.0f);
}

TEST(TensorTest, SubtractionWithScalar){
    Tensor a({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    float scalar = 1.0f;
    Tensor t = a - scalar;

    EXPECT_FLOAT_EQ(t({0,0}), 0.0f);
    EXPECT_FLOAT_EQ(t({0,1}), 1.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 2.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 3.0f);
}

TEST(TensorTest, Multiply){
    Tensor a({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b({2,2}, {2.0f, 3.0f, 4.0f, 5.0f});
    Tensor t = a * b;

    EXPECT_FLOAT_EQ(t({0,0}), 2.0f);
    EXPECT_FLOAT_EQ(t({0,1}), 6.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 12.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 20.0f);
}

TEST(TensorTest, MultiplyWithScalar){
    Tensor a({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    float scalar = 2.0f;
    Tensor t = a * scalar;

    EXPECT_FLOAT_EQ(t({0,0}), 2.0f);
    EXPECT_FLOAT_EQ(t({0,1}), 4.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 6.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 8.0f);
}

TEST(TensorTest, Division){
    Tensor a({2,2}, {1.0f, 4.0f, 12.0f, 24.0f});
    Tensor b({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor t = a / b;

    EXPECT_FLOAT_EQ(t({0,0}), 1.0f);
    EXPECT_FLOAT_EQ(t({0,1}), 2.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 4.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 6.0f);
}

TEST(TensorTest, DivisionWithScalar){
    Tensor a({2,2}, {1.0f, 4.0f, 12.0f, 24.0f});
    float scalar = 2.0f;
    Tensor t = a / scalar;

    EXPECT_FLOAT_EQ(t({0,0}), 0.5f);
    EXPECT_FLOAT_EQ(t({0,1}), 2.0f);
    EXPECT_FLOAT_EQ(t({1,0}), 6.0f);
    EXPECT_FLOAT_EQ(t({1,1}), 12.0f);
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