#include <gtest/gtest.h>
#include "tensor/Tensor.h"

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