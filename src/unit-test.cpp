#include "gtest/gtest.h"
#include "index_succinct.hpp"

#include "sdsl/suffix_trees.hpp"

using csa_type = sdsl::csa_wt_int<>;
using cst_type = sdsl::cst_sct3<csa_type>;
using index_type = index_succinct<cst_type>;

class LMTest : public testing::Test {
protected:  
	virtual void SetUp() {
		col = collection(col_path,false);
		idx = index_type(col,false);
	}
	index_type idx;
	collection col;
	const std::string col_path = "../collections/toy/";
};

TEST_F(LMTest, EnsureZeroLast) {
	 EXPECT_EQ(idx.m_cst.csa.text[idx.m_cst.size()-1],0ULL);
	 EXPECT_EQ(idx.m_cst_rev.csa.text[idx.m_cst_rev.size()-1],0ULL);
}

TEST_F(LMTest, Count ) {
	// forward test
	EXPECT_EQ( sdsl::count(idx.m_cst,{5ULL}), 4ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst,{0ULL}), 1ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst,{5ULL,5ULL}), 2ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst,{5ULL,4ULL,5ULL,5ULL,5ULL}), 1ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst,{5ULL,5ULL,5ULL,5ULL,5ULL}), 0ULL);
	// backward test
	EXPECT_EQ( sdsl::count(idx.m_cst_rev,{5ULL}), 4ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst_rev,{0ULL}), 1ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst_rev,{5ULL,5ULL}), 2ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst_rev,{5ULL,4ULL,5ULL,5ULL,5ULL}), 0ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst_rev,{5ULL,5ULL,5ULL,4ULL,5ULL}), 1ULL);
	EXPECT_EQ( sdsl::count(idx.m_cst_rev,{5ULL,5ULL,5ULL,5ULL,5ULL}), 0ULL);
}

TEST_F(LMTest, BackwardSearchSingleSym ) {
	{
		uint64_t sym = 1;
		size_t lb,rb = 0;
		//forward
		sdsl::backward_search(idx.m_cst.csa,0,idx.m_cst.csa.size()-1,sym,lb,rb);
		EXPECT_EQ(lb,1ULL);
		EXPECT_EQ(rb,4ULL);
		// backward
		sdsl::backward_search(idx.m_cst_rev.csa,0,idx.m_cst.csa.size()-1,sym,lb,rb);
		EXPECT_EQ(lb,1ULL);
		EXPECT_EQ(rb,4ULL);
	}
	{
		uint64_t sym = 4;
		size_t lb,rb = 0;
		//forward
		sdsl::backward_search(idx.m_cst.csa,0,idx.m_cst.csa.size()-1,sym,lb,rb);
		EXPECT_EQ(lb,11ULL);
		EXPECT_EQ(rb,17ULL);
		// backward
		sdsl::backward_search(idx.m_cst_rev.csa,0,idx.m_cst.csa.size()-1,sym,lb,rb);
		EXPECT_EQ(lb,11ULL);
		EXPECT_EQ(rb,17ULL);
	}
}

TEST_F(LMTest, vocab_size ) {
	EXPECT_EQ( idx.vocab_size() , 7ULL );
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

