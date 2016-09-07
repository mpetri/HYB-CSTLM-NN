#include "gtest/gtest.h"

#include "knm.hpp"
#include "logging.hpp"
#include "index_types.hpp"

#include <unordered_set>

using namespace cstlm;

typedef testing::Types<cstlm::index_succinct<cstlm::default_cst_int_type> > Implementations;

typedef testing::Types<cstlm::index_succinct<default_cst_byte_type>,
    cstlm::index_succinct<cstlm::default_cst_int_type> > AllImplementations;

struct triplet {
    std::vector<uint64_t> pattern;
    int order;
    double perplexity;
};

// helper function to hash tuples
template <class T>
struct vector_hasher {
    size_t operator()(const std::vector<T>& vec) const
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
    virtual void SetUp()
    {
        {
            if (std::is_same<t_idx, index_succinct<default_cst_byte_type> >::value == true) {
                col = collection(col_path, alphabet_type::byte_alphabet);
            }
            else {
                col = collection(col_path, alphabet_type::word_alphabet);
            }
            idx = t_idx(col, true);
            idx.print_params(true, 10);
        }
    }
    t_idx idx;
    collection col;
    const char* col_path = "../collections/unittest/";
};

template <class t_idx>
class LMPPxTest : public testing::Test {
protected:
    const char* srilm_path = "../UnitTestData/srilm_output/output_srilm_kn";
    const char* kenlm_mkn_path = "../UnitTestData/kenlm_output/output_kenlm";
    std::vector<triplet> srilm_triplets, kenlm_triplets_mkn;

    void load_pplx_triplets(const std::string& path, std::vector<triplet>& out)
    {
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<std::string> x = split(line, '@');
            triplet tri;
            std::vector<std::string> x2 = split(x[0], ' ');
            std::vector<uint64_t> pattern;
            for (std::string word : x2) {
                pattern.push_back(idx.vocab.token2id(word, UNKNOWN_SYM));
            }
            tri.pattern = pattern;
            tri.order = std::stoi(x[1]);
            tri.perplexity = std::stod(x[2]);
            out.push_back(tri);
        }
    }

    virtual void SetUp()
    {
        // std::cout << "CONSTRUCTING LMTest: SetUp() for object " << (void*) this
        // << std::endl;
        {
            col = collection(col_path, alphabet_type::word_alphabet);
            idx = t_idx(col, true);
            idx.print_params(true, 10);
        }

        load_pplx_triplets(srilm_path, srilm_triplets);
        load_pplx_triplets(kenlm_mkn_path, kenlm_triplets_mkn);
    }
    t_idx idx;
    collection col;
    const char* col_path = "../collections/unittest/";
};

TYPED_TEST_CASE(LMPPxTest, Implementations);
TYPED_TEST_CASE(LMTest, AllImplementations);

TYPED_TEST(LMTest, PrecomputedStats_nX)
{
    using pattern_type = typename decltype(this->idx)::pattern_type;
    using value_type = typename pattern_type::value_type;
    pattern_type text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
        std::back_inserter(text));

    /* count the number of ngrams without sentinals */
    for (size_t cgram = 2; cgram <= this->idx.discounts.max_ngram_count;
         cgram++) {
        std::unordered_map<std::vector<value_type>, uint64_t, vector_hasher<value_type> >
            ngram_counts;
        /* compute c-gram stats */
        // -3 to ignore the last three symbols in the collection: UNK EOS EOF
        for (size_t i = 0; i < (text.size() - 3) - (cgram - 1); i++) {
            pattern_type cur_gram(cgram);
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
            if (std::none_of(cng.cbegin(), cng.cend(),
                    [](value_type i) { return i == EOS_SYM; })
                && std::none_of(cng.cbegin(), cng.cend(),
                       [](value_type i) { return i == EOF_SYM; })) {
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
        EXPECT_EQ(act_n1, this->idx.discounts.counts.n1[cgram])
            << "n1[" << cgram << "] count incorrect!";
        EXPECT_EQ(act_n2, this->idx.discounts.counts.n2[cgram])
            << "n2[" << cgram << "] count incorrect!";
        EXPECT_EQ(act_n3, this->idx.discounts.counts.n3[cgram])
            << "n3[" << cgram << "] count incorrect!";
        EXPECT_EQ(act_n4, this->idx.discounts.counts.n4[cgram])
            << "n4[" << cgram << "] count incorrect!";
    }
}

TYPED_TEST(LMTest, PrecomputedStats_nX_cnt)
{
    using pattern_type = typename decltype(this->idx)::pattern_type;
    using value_type = typename pattern_type::value_type;
    pattern_type text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
        std::back_inserter(text));

    /* count the number of ngrams without sentinals */
    for (size_t cgram = 2; cgram <= this->idx.discounts.max_ngram_count;
         cgram++) {
        std::unordered_map<std::vector<value_type>, std::unordered_set<value_type>,
            vector_hasher<value_type> > ngram_counts;
        /* compute N1PlusBack c-gram stats */
        // -3 to ignore the last three symbols in the collection: UNK EOS EOF
        for (size_t i = 0; i < (text.size() - 3) - (cgram - 1); i++) {
            std::vector<value_type> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (i > 0 && text[i - 1] != EOS_SYM) {
                auto precending_syms_set = ngram_counts[cur_gram];
                precending_syms_set.insert(text[i - 1]);
                ngram_counts[cur_gram] = precending_syms_set;
            }
            else {
                if (ngram_counts.find(cur_gram) == ngram_counts.end())
                    ngram_counts[cur_gram] = std::unordered_set<value_type>();
            }
        }
        /* compute the nX_cnt counts */
        uint64_t act_n1_cnt = 0;
        uint64_t act_n2_cnt = 0;
        uint64_t act_n3_cnt = 0;
        uint64_t act_n4_cnt = 0;
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            if (std::none_of(cng.cbegin(), cng.cend(),
                    [](value_type i) { return i == EOS_SYM; })
                && std::none_of(cng.cbegin(), cng.cend(),
                       [](value_type i) { return i == EOF_SYM; })) {
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
                }
                else {
                    // special case: ngram starts with PAT_START_SYM
                    size_t cnt = 0;
                    // -3 to ignore the last three symbols in the collection: UNK EOS EOF
                    for (size_t i = 0; i < (text.size() - 3) - (cng.size() - 1); i++) {
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
        EXPECT_EQ(this->idx.discounts.counts.n1_cnt[cgram], act_n1_cnt)
            << "n1_cnt[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.discounts.counts.n2_cnt[cgram], act_n2_cnt)
            << "n2_cnt[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.discounts.counts.n3_cnt[cgram], act_n3_cnt)
            << "n3_cnt[" << cgram << "] count incorrect!";
        EXPECT_EQ(this->idx.discounts.counts.n4_cnt[cgram], act_n4_cnt)
            << "n4_cnt[" << cgram << "] count incorrect!";
    }
}

TYPED_TEST(LMTest, PrecomputedStats_N1DotPlusPlus)
{
    using pattern_type = typename decltype(this->idx)::pattern_type;
    using value_type = typename pattern_type::value_type;
    pattern_type text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
        std::back_inserter(text));
    std::unordered_set<std::vector<value_type>, vector_hasher<value_type> > uniq_bigrams;
    /* compute c-gram stats */
    // -3 to ignore the last three symbols in the collection: UNK EOS EOF
    for (size_t i = 0; i < (text.size() - 3) - 1; i++) {
        pattern_type cur_gram(2);
        auto beg = text.begin() + i;
        std::copy(beg, beg + 2, cur_gram.begin());
        if (std::none_of(cur_gram.cbegin(), cur_gram.cend(),
                [](value_type i) { return i == EOS_SYM; })
            && std::none_of(cur_gram.cbegin(), cur_gram.cend(),
                   [](value_type i) { return i == EOF_SYM; })) {
            uniq_bigrams.insert(cur_gram);
        }
    }
    size_t act_N1plus_dotdot = uniq_bigrams.size();
    /* compare counts */
    EXPECT_EQ(this->idx.discounts.counts.N1plus_dotdot, act_N1plus_dotdot)
        << "N1plus_dotdot count incorrect!";
}

TYPED_TEST(LMTest, PrecomputedStats_N3plus_dot)
{
    using pattern_type = typename decltype(this->idx)::pattern_type;
    using value_type = typename pattern_type::value_type;
    pattern_type text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
        std::back_inserter(text));
    std::unordered_map<value_type, uint64_t> unigram_freqs;
    /* compute c-gram stats */
    // -3 to ignore the last three symbols in the collection: UNK EOS EOF
    for (size_t i = 0; i < (text.size() - 3); i++) {
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
    EXPECT_EQ(this->idx.discounts.counts.N3plus_dot, act_N3plus_dot)
        << "N3plus_dot count incorrect!";
}

TYPED_TEST(LMTest, N1PlusBack)
{
    // (1) get the text
    using pattern_type = typename decltype(this->idx)::pattern_type;
    using value_type = typename pattern_type::value_type;
    pattern_type text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
        std::back_inserter(text));

    // (2) for all n-gram sizes

    for (size_t cgram = 1; cgram <= this->idx.discounts.max_ngram_count + 5;
         cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusBack counts
        std::unordered_map<std::vector<value_type>, std::unordered_set<value_type>,
            vector_hasher<value_type> > ngram_counts;
        /* compute N1PlusBack c-gram stats */
        // -3 to ignore the last three symbols in the collection: UNK EOS EOF
        for (size_t i = 0; i < (text.size() - 3) - (cgram - 1); i++) {
            std::vector<value_type> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (i > 0 && text[i - 1] != EOS_SYM) {
                auto precending_syms_set = ngram_counts[cur_gram];
                precending_syms_set.insert(text[i - 1]);
                ngram_counts[cur_gram] = precending_syms_set;
            }
            else {
                if (ngram_counts.find(cur_gram) == ngram_counts.end())
                    ngram_counts[cur_gram] = std::unordered_set<value_type>();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            auto expected_N1PlusBack_count = ngc.second.size();
            if (std::none_of(cng.cbegin(), cng.cend(),
                    [](value_type i) { return i == EOS_SYM; })
                && std::none_of(cng.cbegin(), cng.cend(),
                       [](value_type i) { return i == EOF_SYM; })
                && std::none_of(cng.cbegin() + 1, cng.cend() - 1,
                       [](value_type i) { return i == PAT_START_SYM; })
                && std::none_of(cng.cbegin() + 1, cng.cend() - 1,
                       [](value_type i) { return i == PAT_END_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb, rb;
                auto cnt = backward_search(this->idx.cst.csa, 0,
                    this->idx.cst.csa.size() - 1, cng.begin(),
                    cng.end(), lb, rb);
                EXPECT_TRUE(cnt > 0);
                if (cnt > 0) {
                    auto actual_count = this->idx.N1PlusBack(this->idx.cst.node(lb, rb),
                        cng.begin(), cng.end());
                    EXPECT_EQ(actual_count, expected_N1PlusBack_count);
                }
            }
        }
    }
}

TYPED_TEST(LMTest, N1PlusFrontBack)
{
    // (1) get the text
    using pattern_type = typename decltype(this->idx)::pattern_type;
    using value_type = typename pattern_type::value_type;
    pattern_type text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
        std::back_inserter(text));

    // (2) for all n-gram sizes
    for (size_t cgram = 1; cgram <= this->idx.discounts.max_ngram_count + 5;
         cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusFrontBack counts
        std::unordered_map<std::vector<value_type>,
            std::unordered_set<std::vector<value_type>, vector_hasher<value_type> >,
            vector_hasher<value_type> > ngram_counts;
        /* compute N1PlusFrontBack c-gram stats */
        // -3 to ignore the last three symbols in the collection: UNK EOS EOF
        for (size_t i = 1; i < (text.size() - 3) - cgram; i++) {
            std::vector<value_type> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (!((text[i - 1] == EOS_SYM) && (text[i + cgram] == EOS_SYM))) {
                auto ctx_set = ngram_counts[cur_gram];
                std::vector<value_type> ctx{ text[i - 1], text[i + cgram] };
                ctx_set.insert(ctx);
                ngram_counts[cur_gram] = ctx_set;
            }
            else {
                if (ngram_counts.find(cur_gram) == ngram_counts.end())
                    ngram_counts[cur_gram] = std::unordered_set<std::vector<value_type>, vector_hasher<value_type> >();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            auto expected_N1PlusFrontBack_count = ngc.second.size();
            if (std::none_of(cng.cbegin(), cng.cend(),
                    [](value_type i) { return i == EOS_SYM; })
                && std::none_of(cng.cbegin(), cng.cend(),
                       [](value_type i) { return i == EOF_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb, rb;
                auto cnt = backward_search(this->idx.cst.csa, 0,
                    this->idx.cst.csa.size() - 1, cng.begin(),
                    cng.end(), lb, rb);

                EXPECT_TRUE(cnt > 0);
                if (cnt > 0) {
                    auto actual_count = this->idx.N1PlusFrontBack(
                        this->idx.cst.node(lb, rb), cng.begin(), cng.end());
                    EXPECT_EQ(actual_count, expected_N1PlusFrontBack_count);
                }
            }
        }
    }
}

TYPED_TEST(LMTest, N1PlusFront)
{
    // (1) get the text
    using pattern_type = typename decltype(this->idx)::pattern_type;
    using value_type = typename pattern_type::value_type;
    pattern_type text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
        std::back_inserter(text));

    // (2) for all n-gram sizes
    for (size_t cgram = 1; cgram <= this->idx.discounts.max_ngram_count + 5;
         cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusFront counts
        std::unordered_map<std::vector<value_type>, std::unordered_set<value_type>,
            vector_hasher<value_type> > ngram_counts;
        /* compute N1PlusFront c-gram stats */
        // -3 to ignore the last three symbols in the collection: UNK EOS EOF
        for (size_t i = 0; i < (text.size() - 3) - cgram; i++) {
            pattern_type cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (text[i + cgram] != EOS_SYM) {
                auto following_syms_set = ngram_counts[cur_gram];
                following_syms_set.insert(text[i + cgram]);
                ngram_counts[cur_gram] = following_syms_set;
            }
            else {
                ngram_counts[cur_gram] = std::unordered_set<value_type>();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            auto expected_N1PlusFront_count = ngc.second.size();
            if (std::none_of(cng.cbegin(), cng.cend(),
                    [](value_type i) { return i == EOS_SYM; })
                && std::none_of(cng.cbegin(), cng.cend(),
                       [](value_type i) { return i == EOF_SYM; })
                && std::none_of(cng.cbegin() + 1, cng.cend() - 1,
                       [](value_type i) { return i == PAT_START_SYM; })
                && std::none_of(cng.cbegin() + 1, cng.cend() - 1,
                       [](value_type i) { return i == PAT_END_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb, rb;
                auto cnt = backward_search(this->idx.cst.csa, 0,
                    this->idx.cst.csa.size() - 1, cng.begin(),
                    cng.end(), lb, rb);
                EXPECT_TRUE(cnt > 0);
                if (cnt > 0) {
                    auto actual_count = this->idx.N1PlusFront(
                        this->idx.cst.node(lb, rb), cng.begin(), cng.end());
                    EXPECT_EQ(actual_count, expected_N1PlusFront_count);
                }
            }
        }
    }
}

TYPED_TEST(LMTest, N123PlusFront)
{
    // (1) get the text
    using pattern_type = typename decltype(this->idx)::pattern_type;
    using value_type = typename pattern_type::value_type;
    pattern_type text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
        std::back_inserter(text));

    // (2) for all n-gram sizes
    for (size_t cgram = 1; cgram <= this->idx.discounts.max_ngram_count + 5;
         cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusFront counts
        typedef std::map<value_type, uint64_t> t_symbol_counts;
        std::unordered_map<std::vector<value_type>, t_symbol_counts,
            vector_hasher<value_type> > ngram_counts;
        /* compute N1PlusFront c-gram stats */
        // -3 to ignore the last three symbols in the collection: UNK EOS EOF
        for (size_t i = 0; i < (text.size() - 3) - cgram;
             i++) { // FIXME: remove -3 and it fails this test twice, leave -3 it
            // fails this test once
            pattern_type cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (text[i + cgram] != EOS_SYM) {
                auto following_syms = ngram_counts[cur_gram];
                following_syms[text[i + cgram]] += 1;
                ngram_counts[cur_gram] = following_syms;
            }
            else {
                ngram_counts[cur_gram] = t_symbol_counts();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            auto expected_stuff = ngc.second;
            uint64_t expected_n1 = 0, expected_n2 = 0, expected_n3p = 0;
            for (auto symbol_count : expected_stuff) {
                if (symbol_count.second == 1)
                    expected_n1 += 1;
                else if (symbol_count.second == 2)
                    expected_n2 += 1;
                else if (symbol_count.second >= 3)
                    expected_n3p += 1;
            }

            if (std::none_of(cng.cbegin(), cng.cend(),
                    [](value_type i) { return i == EOS_SYM; })
                && std::none_of(cng.cbegin(), cng.cend(),
                       [](value_type i) { return i == EOF_SYM; })
                && std::none_of(cng.cbegin() + 1, cng.cend() - 1,
                       [](value_type i) { return i == PAT_START_SYM; })
                && std::none_of(cng.cbegin() + 1, cng.cend() - 1,
                       [](value_type i) { return i == PAT_END_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb, rb;
                auto cnt = backward_search(this->idx.cst.csa, 0,
                    this->idx.cst.csa.size() - 1, cng.begin(),
                    cng.end(), lb, rb);
                EXPECT_TRUE(cnt > 0);
                if (cnt > 0) {
                    uint64_t n1, n2, n3p, n1p;
                    this->idx.N123PlusFront(this->idx.cst.node(lb, rb), cng.begin(),
                        cng.end(), n1, n2, n3p);
                    n1p = this->idx.N1PlusFront(this->idx.cst.node(lb, rb), cng.begin(),
                        cng.end());

                    // LOG(INFO) << "pattern is " <<
                    // this->idx.m_vocab.id2token(cng.begin(), cng.end()) << " === " <<
                    // cng << " (numberised)";
                    EXPECT_EQ(n1, expected_n1);
                    EXPECT_EQ(n2, expected_n2);
                    EXPECT_EQ(n3p, expected_n3p);
                    EXPECT_EQ(n1p, n1 + n2 + n3p);
                }
            }
        }
    }
}

#if 0
TYPED_TEST(LMTest, N123PlusBack)
{
    // (1) get the text
    std::vector<uint64_t> text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
              std::back_inserter(text));

    // (2) for all n-gram sizes
    for (size_t cgram = 1; cgram <= this->idx.discounts.max_ngram_count + 5; cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusFront counts
        typedef std::map<uint64_t, uint64_t> t_symbol_counts;
        std::unordered_map<std::vector<uint64_t>, t_symbol_counts, uint64_vector_hasher> ngram_counts;
        /* compute N1PlusFront c-gram stats */
        // -3 to ignore the last three symbols in the collection: UNK EOS EOF
        for (size_t i = 0; i < (text.size() - 3) - cgram; i++) { //FIXME: remove -3 and it fails this test twice, leave -3 it fails this test once
            std::vector<uint64_t> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (i > 0 && text[i - 1] != EOS_SYM) {
                auto preceeding_syms = ngram_counts[cur_gram];
                preceeding_syms[text[i - 1]] += 1;
                ngram_counts[cur_gram] = preceeding_syms;
            } else {
                ngram_counts[cur_gram] = t_symbol_counts();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            uint64_t expected_n1 = 0, expected_n2 = 0, expected_n3p = 0;
            for (auto symbol_count : ngc.second) {
                if (symbol_count.second == 1)
                    expected_n1 += 1;
                else if (symbol_count.second == 2)
                    expected_n2 += 1;
                else if (symbol_count.second >= 3)
                    expected_n3p += 1;
            }

            if (std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOS_SYM; })
                && std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOF_SYM; })
                && std::none_of(cng.cbegin() + 1, cng.cend() - 1, [](uint64_t i) { return i == PAT_START_SYM; })
                && std::none_of(cng.cbegin() + 1, cng.cend() - 1, [](uint64_t i) { return i == PAT_END_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb, rb;
                auto cnt = backward_search(this->idx.cst.csa, 0, this->idx.cst.csa.size() - 1,
                                           cng.begin(), cng.end(), lb, rb);
                EXPECT_TRUE(cnt > 0);
                if (cnt > 0) {
                    uint64_t n1, n2, n3p, n1p;
                    this->idx.N123PlusBack(this->idx.cst.node(lb, rb), cng.begin(), cng.end(), n1, n2, n3p);
                    n1p = this->idx.N1PlusBack(this->idx.cst.node(lb, rb), cng.begin(), cng.end());

                    //LOG(INFO) << "pattern is " << this->idx.m_vocab.id2token(cng.begin(), cng.end());
                    EXPECT_EQ(n1, expected_n1);
                    EXPECT_EQ(n2, expected_n2);
                    EXPECT_EQ(n3p, expected_n3p);
                    EXPECT_EQ(n1p, n1 + n2 + n3p);
                }
            }
        }
    }
}

TYPED_TEST(LMTest, N123PlusFrontBack)
{
    // (1) get the text
    std::vector<uint64_t> text;
    std::copy(this->idx.cst.csa.text.begin(), this->idx.cst.csa.text.end(),
              std::back_inserter(text));

    // (2) for all n-gram sizes
    for (size_t cgram = 1; cgram <= this->idx.discounts.max_ngram_count + 5; cgram++) {
        // (3) determine all valid ngrams and their actual N1PlusFrontBack counts
        typedef std::map<std::pair<uint64_t, uint64_t>, uint64_t> t_symbol_counts;
        std::unordered_map<std::vector<uint64_t>, t_symbol_counts, uint64_vector_hasher> ngram_counts;
        /* compute N1PlusFrontBack c-gram stats */
        // -3 to ignore the last three symbols in the collection: UNK EOS EOF
        for (size_t i = 1; i < (text.size() - 3) - cgram; i++) {
            std::vector<uint64_t> cur_gram(cgram);
            auto beg = text.begin() + i;
            std::copy(beg, beg + cgram, cur_gram.begin());

            if (!((text[i - 1] == EOS_SYM) && (text[i + cgram] == EOS_SYM))) {
                auto ctx_counts = ngram_counts[cur_gram];
                std::pair<uint64_t, uint64_t> ctx(text[i - 1], text[i + cgram]);
                ctx_counts[ctx] += 1;
                ngram_counts[cur_gram] = ctx_counts;
            } else {
                if (ngram_counts.find(cur_gram) == ngram_counts.end())
                    ngram_counts[cur_gram] = t_symbol_counts();
            }
        }

        // (4) for all valid ngrams, query the index
        for (const auto& ngc : ngram_counts) {
            const auto& cng = ngc.first;
            uint64_t expected_n1 = 0, expected_n2 = 0, expected_n3p = 0;
            for (const auto& item : ngc.second) {
                if (item.second == 1)
                    expected_n1 += 1;
                else if (item.second == 2)
                    expected_n2 += 1;
                else
                    expected_n3p += 1;
            }

            if (std::none_of(cng.cbegin(), cng.cend(), [](uint64_t i) { return i == EOS_SYM; })
                && std::none_of(cng.cbegin(), cng.cend(),
                                [](uint64_t i) { return i == EOF_SYM; })) {
                // (1) perform backward search on reverse csa to get the node [lb,rb]
                uint64_t lb, rb;
                auto cnt = backward_search(this->idx.cst.csa, 0, this->idx.cst.csa.size() - 1,
                                           cng.begin(), cng.end(), lb, rb);

                EXPECT_TRUE(cnt > 0);
                if (cnt > 0) {
                    uint64_t actual_n1, actual_n2, actual_n3p, actual_n1p;
                    this->idx.N123PlusFrontBack(this->idx.cst.node(lb, rb),
                                                cng.begin(), cng.end(),
                                                actual_n1, actual_n2, actual_n3p);
                    actual_n1p = this->idx.N1PlusFrontBack(this->idx.cst.node(lb, rb), cng.begin(), cng.end());

                    //LOG(INFO) << "pattern is " << this->idx.m_vocab.id2token(cng.begin(), cng.end());
                    EXPECT_EQ(actual_n1, expected_n1);
                    EXPECT_EQ(actual_n2, expected_n2);
                    EXPECT_EQ(actual_n3p, expected_n3p);
                    EXPECT_EQ(actual_n1 + actual_n2 + actual_n3p, actual_n1p);
                }
            }
        }
    }
}
#endif
// checks whether perplexities match
// precision of comparison is set to 1e-4
TYPED_TEST(LMPPxTest, Perplexity)
{
    for (unsigned int i = 0; i < this->srilm_triplets.size(); i++) {
        auto srilm = this->srilm_triplets[i];
        double perplexity = sentence_perplexity_kneser_ney(
            this->idx, srilm.pattern, srilm.order, false);
        EXPECT_NEAR(perplexity, srilm.perplexity, 1e-4);
    }
}

TYPED_TEST(LMPPxTest, PerplexityMKN)
{
    int last_order = -1;
    for (unsigned int i = 0; i < this->kenlm_triplets_mkn.size(); i++) {
        auto kenlm = this->kenlm_triplets_mkn[i];
        last_order = kenlm.order;
        double perplexity = sentence_perplexity_kneser_ney(
            this->idx, kenlm.pattern, kenlm.order, true);
        EXPECT_NEAR(perplexity, kenlm.perplexity, 1e-2);
    }
}

int main(int argc, char* argv[])
{
    enable_logging = true;

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
