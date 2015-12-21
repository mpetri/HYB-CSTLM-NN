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
#include "kn.hpp"
#include "kn_modified.hpp"
#include "query.hpp"
#include "query_kn.hpp"

template <class t_idx, class t_pattern>
double sentence_logprob_kneser_ney(const t_idx& idx, const t_pattern& word_vec,
                                   uint64_t& /*M*/, uint64_t ngramsize,
                                   bool ismkn, bool isfishy)
{
    if (ismkn) {
        double final_score = 0;
        LMQueryMKN<t_idx,typename t_pattern::value_type> query(&idx, ngramsize);
        for (const auto& word : word_vec)
            final_score += log10(query.append_symbol(word));
        return final_score;
    } else {
        double final_score = 0;
        LMQueryKN<t_idx,typename t_pattern::value_type> query(&idx, ngramsize);
        for (const auto& word : word_vec)
            final_score += log10(query.append_symbol(word));
        return final_score;
    }

    // LOG(INFO) << "sentence_logprob_kneser_ney for: " <<
    // idx.m_vocab.id2token(word_vec.begin(), word_vec.end());
    // LOG(INFO) << "\tfast: " << fast_index << " mkn: " << ismkn;
    double final_score = 0;
    std::deque<uint64_t> pattern_deq;
    for (const auto& word : word_vec) {
        pattern_deq.push_back(word);
        if (word == PAT_START_SYM)
            continue;
        if (pattern_deq.size() > ngramsize) {
            pattern_deq.pop_front();
        }
        std::vector<uint64_t> pattern(pattern_deq.begin(), pattern_deq.end());
        /*
    if (pattern.back() == UNKNOWN_SYM) {
        M = M - 1; // excluding OOV from perplexity - identical to SRILM ppl
    }
*/
        double score;
        if (!ismkn)
            score = prob_kneser_ney(idx, pattern.begin(), pattern.end(), ngramsize);
        else
            score = prob_mod_kneser_ney(idx, pattern.begin(), pattern.end(),
                                        ngramsize, isfishy);
        final_score += log10(score);
    }
    // LOG(INFO) << "sentence_logprob_kneser_ney returning: " << final_score;
    return final_score;
}

template <class t_idx, class t_pattern>
double sentence_perplexity_kneser_ney(const t_idx& idx, t_pattern& pattern,
                                      uint32_t ngramsize, bool ismkn,
                                      bool isfishy)
{
    auto pattern_size = pattern.size();
    pattern.push_back(PAT_END_SYM);
    pattern.insert(pattern.begin(), PAT_START_SYM);
    // run the query
    uint64_t M = pattern_size + 1;
    double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize, ismkn, isfishy);
    double perplexity = pow(10, -(1 / (double)M) * sentenceprob);
    return perplexity;
}

// required by Moses
template <class t_idx, class t_pattern>
uint64_t patternId(const t_idx& idx, const t_pattern& word_vec)
{
    uint64_t lb = 0, rb = idx.m_cst.size() - 1;
    backward_search(idx.m_cst.csa, lb, rb, word_vec.begin(), word_vec.end(), lb,
                    rb);
    auto node = idx.m_cst.node(lb, rb);
    return idx.m_cst.id(node);
}
