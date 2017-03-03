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

#include "common.hpp"

namespace cstlm {


template <class t_idx, class t_pattern>
sentence_eval sentence_logprob_kneser_ney(const t_idx&     idx,
                                          const t_pattern& word_vec,
                                          uint64_t& /*M*/,
                                          uint64_t ngramsize,
                                          bool     ismkn,
                                          bool     use_cache)
{
    double            final_score = 0;
    size_t            num_tokens  = 0;
    LMQueryMKN<t_idx> query(&idx, ngramsize, true, use_cache);
    bool first = true;
    for (const auto& word : word_vec) {
        auto score = query.append_symbol(word.big_id);
        if (!first && word.small_id != UNKNOWN_SYM && word.big_id != UNKNOWN_SYM) {
            final_score += score;
            num_tokens++;
        }
	first = false;
    }
    return sentence_eval(final_score, num_tokens);
}

template <class t_idx, class t_pattern>
sentence_eval sentence_logprob_kneser_ney2(const t_idx&     idx,
                                          const t_pattern& word_vec,
                                          uint64_t& /*M*/,
                                          uint64_t ngramsize,
                                          bool     ismkn,
                                          bool     use_cache)
{
    double            final_score = 0;
    size_t            num_tokens  = 0;
    LMQueryMKN<t_idx> query(&idx, ngramsize, true, use_cache);
    bool first = true;
    for (const auto& word : word_vec) {
        auto score = query.append_symbol(word.big_id);
        if(!first && word.big_id != UNKNOWN_SYM) {
		num_tokens++;
        	final_score += score;
	}
	first = false;
    }
    return sentence_eval(final_score, num_tokens);
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
    double perplexity   = exp(-(1 / (double)M) * sentenceprob);
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
