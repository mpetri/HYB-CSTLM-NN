#include "gtest/gtest.h"
#include "index_succinct.hpp"
#include "sdsl/suffix_trees.hpp"
#include "query-index-knm.hpp"

using csa_type = sdsl::csa_wt_int<>;
using cst_type = sdsl::cst_sct3<csa_type>;
using index_type = index_succinct<cst_type>;

struct triplet{
    std::vector<uint64_t> pattern; 
	int order;
	double perplexity;
};


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

class LMTest : public testing::Test {
protected:  
	const std::string srilm_path = "../UnitTestData/srilm_output/output";
	std::vector<triplet> sdsl_triplets;
	std::vector<triplet> srilm_triplets;
  	virtual void SetUp() {
		{
			col = collection(col_path,false);
			idx = index_type(col,false);
		}

		{
			std::ifstream file(srilm_path);
                	std::string line;
                	while (std::getline(file, line)) {
                        	std::istringstream iss(line);
							std::vector<std::string> x = split(line, '@');
                            triplet tri;
                            std::vector<std::string> x2 = split(x[0],' ');
							std::vector<uint64_t> pattern;
							for(std::string word : x2)
							{
								pattern.push_back(std::stoi(word));
							}
							tri.pattern = pattern;
                            tri.order = std::stoi(x[1]);
                            tri.perplexity = std::stod(x[2]);
                            srilm_triplets.push_back(tri);
                	}
		}
		
    }
	index_type idx;
	collection col;
	const std::string col_path = "../collections/unittest/";
};

// checks whether perplexities match
// precision of comparison is set to 1e-4
TEST_F(LMTest, Perplexity) {
	for(unsigned int i=0;i<srilm_triplets.size();i++)
	{
		triplet srilm = srilm_triplets[i];
		double perplexity = gate(idx, srilm.pattern, srilm.order, false);
	//	cout<<"order "<<srilm.order<<" perplexity-srilm "<<srilm.perplexity<<" perplexity-sdsl "<<perplexity<<endl;
		EXPECT_NEAR(perplexity, srilm.perplexity, 1e-4);
	}
}

int main(int argc, char* argv[])
{

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

