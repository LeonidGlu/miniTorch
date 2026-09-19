#include <gtest/gtest.h>
#include "tensor/Tensor.h"

TEST(TensorIndexingTest, Access){
    Tensor t({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
    t({0,0}) = 10.0f;
    
    EXPECT_FLOAT_EQ(t({0,0}), 10.0f);
}