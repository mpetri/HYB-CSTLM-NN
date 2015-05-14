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

// Computes the probability of P( x | a b c ... ) using raw occurrence counts.
// Note that the backoff probability uses the lower order variants of this method.
//      idx -- the index
//      pattern -- iterators into pattern (is this in order or reversed order???)
//      lb, rb -- left and right bounds on the forward CST (spanning the full index for this method???)
template <class t_idx>
double highestorder(const t_idx& idx, uint64_t level, const bool unk,
                    std::vector<uint64_t>::const_iterator pattern_begin,
                    std::vector<uint64_t>::const_iterator pattern_end,
                    uint64_t& lb, uint64_t& rb,
                    uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d,
                    uint64_t ngramsize)
{
    double backoff_prob = pkn(idx, level, unk,
                              pattern_begin + 1, pattern_end,
                              lb, rb,
                              lb_rev, rb_rev, char_pos, d, ngramsize);
    auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
    uint64_t denominator = 0;
    uint64_t c = 0;

    if (forward_search(idx.m_cst_rev, node, d, *pattern_begin, char_pos) > 0) {
        lb_rev = idx.m_cst_rev.lb(node);
        rb_rev = idx.m_cst_rev.rb(node);
        c = rb_rev - lb_rev + 1;
    }
    uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
    double D = 0;
    if (pattern_size == ngramsize)
        D = idx.discount(ngramsize);
    else
        //which is the special case of n<ngramsize that starts with <s>
        D = idx.discount(pattern_size, true);
    double numerator = 0;
    if (!unk && c - D > 0) {
        numerator = c - D;
    }

    uint64_t N1plus_front = 0;
    if (backward_search(idx.m_cst.csa, lb, rb, *pattern_begin, lb, rb) > 0) {
        denominator = rb - lb + 1;
        N1plus_front = idx.N1PlusFront(lb, rb, pattern_begin, pattern_end - 1);
    } else {
        return backoff_prob;
    }

    double output = (numerator / denominator) + (D * N1plus_front / denominator) * backoff_prob;
    return output;
}

template <class t_idx>
double lowerorder(const t_idx& idx, uint64_t level, const bool unk,
                  std::vector<uint64_t>::const_iterator pattern_begin,
                  std::vector<uint64_t>::const_iterator pattern_end,
                  uint64_t& lb, uint64_t& rb,
                  uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d, uint64_t ngramsize)
{
    level = level - 1;
    double backoff_prob = pkn(idx, level, unk,
                              pattern_begin + 1, pattern_end,
                              lb, rb,
                              lb_rev, rb_rev, char_pos, d, ngramsize);

    uint64_t c = 0;
    auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
    if (forward_search(idx.m_cst_rev, node, d, *(pattern_begin), char_pos) > 0) {
        lb_rev = idx.m_cst_rev.lb(node);
        rb_rev = idx.m_cst_rev.rb(node);
        c = idx.N1PlusBack(lb_rev, rb_rev, pattern_begin, pattern_end);
    }

    double D = idx.discount(level, true);
    double numerator = 0;
    if (!unk && c - D > 0) {
        numerator = c - D;
    }

    if (backward_search(idx.m_cst.csa, lb, rb, *(pattern_begin), lb, rb) > 0) { //TODO CHECK: what happens to the bounds when this is false?
        auto N1plus_front = idx.N1PlusFront(lb, rb, pattern_begin, pattern_end - 1);
        auto back_N1plus_front = idx.N1PlusFrontBack(lb, rb, lb_rev, rb_rev, pattern_begin, pattern_end - 1);
        // FIXME: for the index_succinct version of N1PlusFrontBack this call above can be
        // avoided for patterns that begin with <s> and/or end with </s> using 'N1PlusFront' and 'c'
        // But might not be worth bothering, as these counts are stored explictly for the
        // faster version of the code so there would be no win here.
        d++;
        return (numerator / back_N1plus_front) + (D * N1plus_front / back_N1plus_front) * backoff_prob;
    } else {
        return backoff_prob;
    }
}

template <class t_idx>
double lowestorder(const t_idx& idx,
                   std::vector<uint64_t>::const_iterator pattern_begin,
                   std::vector<uint64_t>::const_iterator pattern_end,
                   uint64_t& lb_rev, uint64_t& rb_rev,
                   uint64_t& char_pos, uint64_t& d)
{
    auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
    double denominator = 0;
    forward_search(idx.m_cst_rev, node, d, *pattern_begin, char_pos);
    d++;
    denominator = idx.m_precomputed.N1plus_dotdot;
    lb_rev = idx.m_cst_rev.lb(node);
    rb_rev = idx.m_cst_rev.rb(node);
    int numerator = idx.N1PlusBack(lb_rev, rb_rev, pattern_begin, pattern_end); //TODO precompute this
    double probability = (double)numerator / denominator;
    return probability;
}

//special lowest order handler for P_{KN}(unknown)
template <class t_idx>
double lowestorder_unk(const t_idx& idx)
{
    double denominator = idx.m_precomputed.N1plus_dotdot;
    double probability = idx.discount(1, true) / denominator;
    return probability;
}

template <class t_idx>
double pkn(const t_idx& idx, uint64_t level, const bool unk,
           std::vector<uint64_t>::const_iterator pattern_begin,
           std::vector<uint64_t>::const_iterator pattern_end,
           uint64_t& lb, uint64_t& rb,
           uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d, uint64_t ngramsize)
{
    uint64_t size = std::distance(pattern_begin, pattern_end);
    double probability = 0;
    if ((size == ngramsize && ngramsize != 1) || (*pattern_begin == PAT_START_SYM)) {
        probability = highestorder(idx, level, unk,
                                   pattern_begin, pattern_end,
                                   lb, rb,
                                   lb_rev, rb_rev, char_pos, d, ngramsize);
    } else if (size < ngramsize && size != 1) {
        if (size == 0)
            exit(1);

        probability = lowerorder(idx, level, unk,
                                 pattern_begin, pattern_end,
                                 lb, rb,
                                 lb_rev, rb_rev, char_pos, d, ngramsize);

    } else if (size == 1 || ngramsize == 1) {
        if (!unk) {
            probability = lowestorder(idx, pattern_end - 1, pattern_end,
                                      lb_rev, rb_rev, char_pos, d);
        } else {
            probability = lowestorder_unk(idx);
        }
    }
    return probability;
}

template <class t_idx>
double run_query_knm(const t_idx& idx, const std::vector<uint64_t>& word_vec, uint64_t& M, uint64_t ngramsize)
{
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
        uint64_t lb_rev = 0, rb_rev = idx.m_cst_rev.size() - 1, lb = 0, rb = idx.m_cst.size() - 1;
        uint64_t char_pos = 0, d = 0;
        int size = std::distance(pattern.begin(), pattern.end());
        bool unk = false;
        if (pattern.back() == 77777) {
            unk = true;
            M = M - 1; // excluding OOV from perplexity - identical to SRILM ppl
        }
        double score = pkn(idx, size, unk,
                           pattern.begin(), pattern.end(),
                           lb, rb,
                           lb_rev, rb_rev, char_pos, d, ngramsize);
        final_score += log10(score);
    }
    return final_score;
}

template <class t_idx>
double gate(const t_idx& idx, std::vector<uint64_t> pattern, uint32_t ngramsize)
{
    auto pattern_size = pattern.size();
    std::string pattern_string;
    pattern.push_back(PAT_END_SYM);
    pattern.insert(pattern.begin(), PAT_START_SYM);
    // run the query
    uint64_t M = pattern_size + 1;
    double sentenceprob = run_query_knm(idx, pattern, M, ngramsize);
    double perplexity = pow(10, -(1 / (double)M) * sentenceprob);
    return perplexity;
}
