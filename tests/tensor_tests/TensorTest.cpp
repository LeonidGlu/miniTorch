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