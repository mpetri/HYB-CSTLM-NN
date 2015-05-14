#include "gtest/gtest.h"

#include "knm.hpp"
#include "logging.hpp"
#include "index_types.hpp"

#include <unordered_set>

typedef testing::Types<index_succinct<default_cst_type>,
                       index_succinct_store_n1fb<default_cst_type> > Implementations;

struct triplet {
    std::vector<uint64_t> pattern;
    int order;
    double perplexity;
};

// helper function to hash tuples
struct uint64_vector_hasher {
    size_t operator()(const std::vector<uint64_t>& vec) const
    {
        std::size_t seed = 0;
        for (auto& i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
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

template <class t_idx>
class LMTest : public testing::Test {
protected:
    const char* srilm_path = "../UnitTestData/srilm_output/output";
    std::vector<triplet> sdsl_triplets;
    std::vector<triplet> srilm_triplets;
    virtual void SetUp()
    {
        //std::cout << "CONSTRUCTING LMTest: SetUp() for object " << (void*) this << std::endl;
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
    const char* col_path = "../collections/unittest/";
};

TYPED_TEST_CASE(LMTest, Implementations);

TYPED_TEST(LMTest, PrecomputedStats_nX)
{
    std::vector<uint64_t> text;
    std::copy(this->idx.m_cst.csa.text.begin(), this->idx.m_cst.csa.text.end(), std::back_inserter(text));

    /* count the number of ngrams without sentinals */
    for (size_t cgram = 2; cgram <= this->idx.m_precomputed.max_ngram_count; cgram++) {
        std::unordered_map<std::vector<uint64_t>, uint64_t, uint64_vector_hasher> ngram_counts;
        /* compute c-gram stats */
        for (size_t i = 0; i < text.size() - (cgram - 1); i++) {
            std::vector<uint64_t> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());
            ngram_counts[cur_gram] += 1;
        }
        /* compute the nX counts */
        uint64_t act_n1 = 0;
        uint64_t act_n2 = 0;
        uint64_t act_n3 = 0;
        uint64_t act_n4 = 0;
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            if (std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOS_SYM; }) && std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOF_SYM; })) {
                auto cnt = ngc.second;
                switch (cnt) {
                case 1:
                    act_n1++;
                    break;
                case 2:
                    act_n2++;
                    break;
                case 3:
                    act_n3++;
                    break;
                case 4:
                    act_n4++;
                    break;
                }
            }
        }
        /* compare counts */
        EXPECT_EQ(this->idx.m_precomputed.n1[cgram], act_n1) << "n1[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.m_precomputed.n2[cgram], act_n2) << "n2[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.m_precomputed.n3[cgram], act_n3) << "n3[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.m_precomputed.n4[cgram], act_n4) << "n4[" << cgram << "] count incorrect!";
    }
}

TYPED_TEST(LMTest, PrecomputedStats_nX_cnt)
{
    std::vector<uint64_t> text;
    std::copy(this->idx.m_cst.csa.text.begin(), this->idx.m_cst.csa.text.end(), std::back_inserter(text));

    /* count the number of ngrams without sentinals */
    for (size_t cgram = 2; cgram <= this->idx.m_precomputed.max_ngram_count; cgram++) {
        std::unordered_map<std::vector<uint64_t>, std::unordered_set<uint64_t>, uint64_vector_hasher> ngram_counts;
        /* compute N1PlusBack c-gram stats */
        for (size_t i = 0; i < text.size() - (cgram - 1); i++) {
            std::vector<uint64_t> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (i > 0 && text[i - 1] != EOS_SYM) {
                auto precending_syms_set = ngram_counts[cur_gram];
                precending_syms_set.insert(text[i - 1]);
                ngram_counts[cur_gram] = precending_syms_set;
            } else {
                if (ngram_counts.find(cur_gram) == ngram_counts.end())
                    ngram_counts[cur_gram] = std::unordered_set<uint64_t>();
            }
        }
        /* compute the nX_cnt counts */
        uint64_t act_n1_cnt = 0;
        uint64_t act_n2_cnt = 0;
        uint64_t act_n3_cnt = 0;
        uint64_t act_n4_cnt = 0;
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            if (std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOS_SYM; }) && std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOF_SYM; })) {
                if (cng[0] != PAT_START_SYM) {
                    auto cnt = ngc.second.size();
                    switch (cnt) {
                    case 1:
                        act_n1_cnt++;
                        break;
                    case 2:
                        act_n2_cnt++;
                        break;
                    case 3:
                        act_n3_cnt++;
                        break;
                    case 4:
                        act_n4_cnt++;
                        break;
                    }
                } else {
                    // special case: ngram starts with PAT_START_SYM
                    size_t cnt = 0;
                    for (size_t i = 0; i < text.size() - (cng.size() - 1); i++) {
                        if (std::equal(cng.begin(), cng.end(), text.begin() + i)) {
                            cnt++;
                        }
                    }
                    switch (cnt) {
                    case 1:
                        act_n1_cnt++;
                        break;
                    case 2:
                        act_n2_cnt++;
                        break;
                    case 3:
                        act_n3_cnt++;
                        break;
                    case 4:
                        act_n4_cnt++;
                        break;
                    }
                }
            }
        }
        /* compare counts */
        EXPECT_EQ(this->idx.m_precomputed.n1_cnt[cgram], act_n1_cnt) << "n1_cnt[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.m_precomputed.n2_cnt[cgram], act_n2_cnt) << "n2_cnt[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.m_precomputed.n3_cnt[cgram], act_n3_cnt) << "n3_cnt[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.m_precomputed.n4_cnt[cgram], act_n4_cnt) << "n4_cnt[" << cgram << "] count incorrect!";
    }
}

TYPED_TEST(LMTest, PrecomputedStats_N1DotPlusPlus)
{
    std::vector<uint64_t> text;
    std::copy(this->idx.m_cst.csa.text.begin(), this->idx.m_cst.csa.text.end(), std::back_inserter(text));
    std::unordered_set<std::vector<uint64_t>, uint64_vector_hasher> uniq_bigrams;
    /* compute c-gram stats */
    for (size_t i = 0; i < text.size() - 1; i++) {
        std::vector<uint64_t> cur_gram(2);
        auto beg = text.begin() + i;
        std::copy(beg, beg + 2, cur_gram.begin());
        if (std::none_of(cur_gram.cbegin(), cur_gram.cend(), [](uint64_t i) { return i == EOS_SYM; }) && std::none_of(cur_gram.cbegin(), cur_gram.cend(), [](uint64_t i) { return i == EOF_SYM; })) {
            uniq_bigrams.insert(cur_gram);
        }
    }
    size_t act_N1plus_dotdot = uniq_bigrams.size();
    /* compare counts */
    EXPECT_EQ(this->idx.m_precomputed.N1plus_dotdot, act_N1plus_dotdot) << "N1plus_dotdot count incorrect!";
}

TYPED_TEST(LMTest, PrecomputedStats_N3plus_dot)
{
    std::vector<uint64_t> text;
    std::copy(this->idx.m_cst.csa.text.begin(), this->idx.m_cst.csa.text.end(), std::back_inserter(text));
    std::unordered_map<uint64_t, uint64_t> unigram_freqs;
    /* compute c-gram stats */
    for (size_t i = 0; i < text.size(); i++) {
        auto sym = text[i];
        if (sym != EOS_SYM && sym != EOF_SYM)
            unigram_freqs[sym]++;
    }
    size_t act_N3plus_dot = 0;
    for (const auto& uc : unigram_freqs) {
        if (uc.second >= 3)
            act_N3plus_dot++;
    }
    /* compare counts */
    EXPECT_EQ(this->idx.m_precomputed.N3plus_dot, act_N3plus_dot) << "N3plus_dot count incorrect!";
}

TYPED_TEST(LMTest, N1PlusBack)
{
    // (1) get the text
    std::vector<uint64_t> text;
    std::copy(this->idx.m_cst.csa.text.begin(), this->idx.m_cst.csa.text.end(), std::back_inserter(text));

    // (2) for all n-gram sizes

    for (size_t cgram = 2; cgram <= this->idx.m_precomputed.max_ngram_count; cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusBack counts
        std::unordered_map<std::vector<uint64_t>, std::unordered_set<uint64_t>, uint64_vector_hasher> ngram_counts;
        /* compute N1PlusBack c-gram stats */
        for (size_t i = 0; i < text.size() - (cgram - 1); i++) {
            std::vector<uint64_t> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (i > 0 && text[i - 1] != EOS_SYM) {
                auto precending_syms_set = ngram_counts[cur_gram];
                precending_syms_set.insert(text[i - 1]);
                ngram_counts[cur_gram] = precending_syms_set;
            } else {
                if (ngram_counts.find(cur_gram) == ngram_counts.end())
                    ngram_counts[cur_gram] = std::unordered_set<uint64_t>();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            auto expected_N1PlusBack_count = ngc.second.size();
            if (std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOS_SYM; }) && std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOF_SYM; }) && std::none_of(cng.cbegin() + 1, cng.cend() - 1, [](uint64_t i) { return i == PAT_START_SYM; }) && std::none_of(cng.cbegin() + 1, cng.cend() - 1, [](uint64_t i) { return i == PAT_END_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb_rev, rb_rev;
                auto rev_cnt = backward_search(this->idx.m_cst_rev.csa,
                                               0, this->idx.m_cst_rev.csa.size() - 1,
                                               cng.rbegin(), cng.rend(),
                                               lb_rev, rb_rev);
                EXPECT_TRUE(rev_cnt > 0);
                if (rev_cnt > 0) {
                    auto actual_count = this->idx.N1PlusBack(lb_rev, rb_rev, cng.begin(), cng.end());
                    EXPECT_EQ(actual_count, expected_N1PlusBack_count);
                }
            }
        }
    }
}

TYPED_TEST(LMTest, N1PlusFrontBack)
{
    // (1) get the text
    std::vector<uint64_t> text;
    std::copy(this->idx.m_cst.csa.text.begin(), this->idx.m_cst.csa.text.end(), std::back_inserter(text));

    // (2) for all n-gram sizes
    for (size_t cgram = 2; cgram <= this->idx.m_precomputed.max_ngram_count; cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusFrontBack counts
        std::unordered_map<std::vector<uint64_t>, std::unordered_set<std::vector<uint64_t>, uint64_vector_hasher>,
                           uint64_vector_hasher> ngram_counts;
        /* compute N1PlusFrontBack c-gram stats */
        for (size_t i = 1; i < text.size() - cgram; i++) {
            std::vector<uint64_t> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (!((text[i - 1] == EOS_SYM) && (text[i + cgram] == EOS_SYM))) {
                auto ctx_set = ngram_counts[cur_gram];
                std::vector<uint64_t> ctx{ text[i - 1], text[i + cgram] };
                ctx_set.insert(ctx);
                ngram_counts[cur_gram] = ctx_set;
            } else {
                if (ngram_counts.find(cur_gram) == ngram_counts.end())
                    ngram_counts[cur_gram] = std::unordered_set<std::vector<uint64_t>, uint64_vector_hasher>();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            auto expected_N1PlusFrontBack_count = ngc.second.size();
            if (std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOS_SYM; }) && std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOF_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb, rb;
                auto cnt = backward_search(this->idx.m_cst.csa,
                                           0, this->idx.m_cst.csa.size() - 1,
                                           cng.begin(), cng.end(),
                                           lb, rb);
                EXPECT_TRUE(cnt > 0);
                uint64_t lb_rev, rb_rev;
                auto rev_cnt = backward_search(this->idx.m_cst_rev.csa,
                                               0, this->idx.m_cst_rev.csa.size() - 1,
                                               cng.rbegin(), cng.rend(),
                                               lb_rev, rb_rev);
                EXPECT_TRUE(rev_cnt > 0);
                if (cnt > 0) {
                    auto actual_count = this->idx.N1PlusFrontBack(lb, rb, lb_rev, rb_rev, cng.begin(), cng.end());
                    EXPECT_EQ(actual_count, expected_N1PlusFrontBack_count);
                }
            }
        }
    }
}

TYPED_TEST(LMTest, N1PlusFront)
{
    // (1) get the text
    std::vector<uint64_t> text;
    std::copy(this->idx.m_cst.csa.text.begin(), this->idx.m_cst.csa.text.end(), std::back_inserter(text));

    // (2) for all n-gram sizes
    for (size_t cgram = 2; cgram <= this->idx.m_precomputed.max_ngram_count; cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusFront counts
        std::unordered_map<std::vector<uint64_t>, std::unordered_set<uint64_t>, uint64_vector_hasher> ngram_counts;
        /* compute N1PlusFront c-gram stats */
        for (size_t i = 0; i < text.size() - cgram; i++) {
            std::vector<uint64_t> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (text[i + cgram] != EOS_SYM) {
                auto following_syms_set = ngram_counts[cur_gram];
                following_syms_set.insert(text[i + cgram]);
                ngram_counts[cur_gram] = following_syms_set;
            } else {
                ngram_counts[cur_gram] = std::unordered_set<uint64_t>();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            auto expected_N1PlusFront_count = ngc.second.size();
            if (std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOS_SYM; }) && std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOF_SYM; }) && std::none_of(cng.cbegin() + 1, cng.cend() - 1, [](uint64_t i) { return i == PAT_START_SYM; }) && std::none_of(cng.cbegin() + 1, cng.cend() - 1, [](uint64_t i) { return i == PAT_END_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb, rb;
                auto cnt = backward_search(this->idx.m_cst.csa,
                                           0, this->idx.m_cst.csa.size() - 1,
                                           cng.begin(), cng.end(),
                                           lb, rb);
                EXPECT_TRUE(cnt > 0);
                if (cnt > 0) {
                    auto actual_count = this->idx.N1PlusFront(lb, rb, cng.begin(), cng.end());
                    EXPECT_EQ(actual_count, expected_N1PlusFront_count);
                }
            }
        }
    }
}

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

int main(int argc, char* argv[])
{
    log::start_log(argc, (const char**)argv, false);

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
