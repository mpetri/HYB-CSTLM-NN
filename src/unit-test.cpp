#include "gtest/gtest.h"
#include "index_succinct.hpp"
#include "sdsl/suffix_trees.hpp"

struct triplet{
  	std::string pattern; 
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
	const std::string sdsl_path = "../UnitTestData/sdsl_output/output";
	const std::string srilm_path = "../UnitTestData/srilm_output/output";
	std::vector<triplet> sdsl_triplets;
	std::vector<triplet> srilm_triplets;
  	virtual void SetUp() {
		{
			std::ifstream file(sdsl_path);
    			std::string line;
    			while (std::getline(file, line)) {
        			std::istringstream iss(line);
				std::vector<std::string> x = split(line, '@');
				triplet tri;
				tri.pattern = x[0];
				tri.order = std::stoi(x[1]);
				tri.perplexity = std::stod(x[2]);
				sdsl_triplets.push_back(tri);
    			}
		}
		{
			std::ifstream file(srilm_path);
                	std::string line;
                	while (std::getline(file, line)) {
                        	std::istringstream iss(line);
				std::vector<std::string> x = split(line, '@');
                                triplet tri;
                                tri.pattern = x[0];
                                tri.order = std::stoi(x[1]);
                                tri.perplexity = std::stod(x[2]);
                                srilm_triplets.push_back(tri);
                	}
		}
		
        }
};

// checks the number of lines in the srilm and sdsl 
// outputs that are used for testing
TEST_F(LMTest, OutputsHaveSameSize){
	EXPECT_EQ(srilm_triplets.size(),sdsl_triplets.size());
}


// checks whether the outputs are aligned
// alignment is done based on pattern and ngram-order
TEST_F(LMTest, OutputsAreAlligned){
	for(unsigned int i=0;i<srilm_triplets.size();i++)
	{
		triplet srilm = srilm_triplets[i];
        	triplet sdsl  = sdsl_triplets[i];
		EXPECT_EQ(srilm.order,sdsl.order);
		EXPECT_EQ(srilm.pattern,sdsl.pattern);
	}
}

// checks whether perplexities match
// precision of comparison is set to 1e-4
TEST_F(LMTest, Perplexity) {
	for(unsigned int i=0;i<srilm_triplets.size();i++)
        {
		triplet srilm = srilm_triplets[i];
		triplet sdsl  = sdsl_triplets[i];
		EXPECT_NEAR(sdsl.perplexity, srilm.perplexity, 1e-4);
	}
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

