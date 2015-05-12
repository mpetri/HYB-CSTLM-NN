#include "gtest/gtest.h"
#include "index_succinct.hpp"
#include "sdsl/suffix_trees.hpp"

#include "knm.hpp"

#include "logging.hpp"

using csa_type = sdsl::csa_wt_int<>;
using cst_type = sdsl::cst_sct3<csa_type>;
using index_type = index_succinct<cst_type>;

typedef testing::Types<
       index_succinct<cst_type>
       > Implementations;

struct triplet {
    std::vector<uint64_t> pattern;
    int order;
    double perplexity;
};

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

template<class t_idx>
class LMTest : public testing::Test {
protected:
    const std::string srilm_path = "../UnitTestData/srilm_output/output";
    std::vector<triplet> sdsl_triplets;
    std::vector<triplet> srilm_triplets;
    virtual void SetUp()
    {
        {
            col = collection(col_path);
            idx = t_idx(col);
        }

        {
            std::ifstream file(srilm_path);
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                std::vector<std::string> x = split(line, '@');
                triplet tri;
                std::vector<std::string> x2 = split(x[0], ' ');
                std::vector<uint64_t> pattern;
                for (std::string word : x2) {
                    pattern.push_back(std::stoi(word));
                }
                tri.pattern = pattern;
                tri.order = std::stoi(x[1]);
                tri.perplexity = std::stod(x[2]);
                srilm_triplets.push_back(tri);
            }
        }
    }
    t_idx idx;
    collection col;
    const std::string col_path = "../collections/unittest/";
};

TYPED_TEST_CASE(LMTest, Implementations);

// checks whether perplexities match
// precision of comparison is set to 1e-4
TYPED_TEST(LMTest, Perplexity)
{
    for (unsigned int i = 0; i < this->srilm_triplets.size(); i++) {
        auto srilm = this->srilm_triplets[i];
        double perplexity = gate(this->idx, srilm.pattern, srilm.order);
        EXPECT_NEAR(perplexity, srilm.perplexity, 1e-4);
    }
}

TYPED_TEST(LMTest, N1PlusBack)
{
}

TYPED_TEST(LMTest, discount)
{
}

TYPED_TEST(LMTest, N1PlusFrontBack)
{
}

TYPED_TEST(LMTest, N1PlusFront)
{
}

TYPED_TEST(LMTest, vocab_size)
{
}

int main(int argc,char* argv[])
{
    log::start_log(argc,(const char**)argv,false);

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
