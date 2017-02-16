#pragma once

#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>
#include <iomanip>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"
#include "constants.hpp"
#include "query.hpp"
#include "query_kn.hpp"

namespace cstlm {


template <class t_idx, class t_pattern>
double sentence_logprob_kneser_ney(const t_idx&     idx,
                                   const t_pattern& word_vec,
                                   uint64_t& /*M*/,
                                   uint64_t ngramsize,
                                   bool     ismkn,
                                   bool     use_cache)
{
    if (ismkn) {
        double            final_score = 0;
        LMQueryMKN<t_idx> query(&idx, ngramsize, true, use_cache);
        for (const auto& word : word_vec) {
            final_score += query.append_symbol(word);
            //LOG(INFO) << "\tprob: " << idx.m_vocab.id2token(word) << " is: " << prob;
        }
        //LOG(INFO) << "sentence_logprob_kneser_ney for: "
        //<< idx.m_vocab.id2token(word_vec.begin(), word_vec.end())
        //<< " returning: " << final_score;
        return final_score;
    } else {
        double           final_score = 0;
        LMQueryKN<t_idx> query(&idx, ngramsize);
        for (const auto& word : word_vec)
            final_score += query.append_symbol(word);
        return final_score;
    }
}

template <class t_idx, class t_pattern>
double sentence_perplexity_kneser_ney(
const t_idx& idx, t_pattern& pattern, uint32_t ngramsize, bool ismkn, bool use_cache = true)
{
    auto pattern_size = pattern.size();
    pattern.push_back(PAT_END_SYM);
    pattern.insert(pattern.begin(), PAT_START_SYM);
    // run the query
    uint64_t M          = pattern_size + 1;
    double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize, ismkn, use_cache);
    double perplexity   = pow(10, -(1 / (double)M) * sentenceprob);
    return perplexity;
}

// required by Moses
template <class t_idx, class t_pattern>
uint64_t patternId(const t_idx& idx, const t_pattern& word_vec)
{
    uint64_t lb = 0, rb = idx.cst.size() - 1;
    backward_search(idx.cst.csa, lb, rb, word_vec.begin(), word_vec.end(), lb, rb);
    auto node = idx.cst.node(lb, rb);
    return idx.cst.id(node);
}
}