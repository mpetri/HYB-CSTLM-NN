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


template <class t_idx, class t_pat_iter>
double prob_kneser_ney(const t_idx& idx, t_pat_iter pattern_begin, 
        t_pat_iter pattern_end, uint64_t ngramsize);

template <class t_idx, class t_pattern>
double sentence_logprob_kneser_ney(const t_idx& idx, const t_pattern& word_vec, uint64_t& M, uint64_t ngramsize, 
        bool fast_index=true)
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

        if (pattern.back() == 77777) { // TODO we have an UNK_SYM sentinel, why not use it?
            //unk = true;
            M = M - 1; // excluding OOV from perplexity - identical to SRILM ppl
        }
        double score;
        if (fast_index)
            score = prob_kneser_ney_forward(idx, pattern.begin(), pattern.end(), ngramsize);
        else
            score = prob_kneser_ney(idx, pattern.begin(), pattern.end(), ngramsize);
        final_score += log10(score);

    }
    return final_score;
}

template <class t_idx, class t_pattern>
double sentence_perplexity_kneser_ney(const t_idx& idx, t_pattern &pattern, uint32_t ngramsize)
{
    auto pattern_size = pattern.size();
    pattern.push_back(PAT_END_SYM);
    pattern.insert(pattern.begin(), PAT_START_SYM);
    // run the query
    uint64_t M = pattern_size + 1;
    double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize);
    double perplexity = pow(10, -(1 / (double)M) * sentenceprob);
    return perplexity;
}

// Returns the Kneser-Ney probability of the n-gram defined
// by [pattern_begin, pattern_end) where the last value is being
// predicted given the previous values in the pattern.
template <class t_idx, class t_pat_iter>
double prob_kneser_ney(const t_idx& idx, t_pat_iter pattern_begin, 
        t_pat_iter pattern_end, uint64_t ngramsize)
{
    typedef typename t_idx::cst_type::node_type t_node;
    double probability = 1.0;
    t_node node = idx.m_cst.root();
    t_node node_rev = idx.m_cst_rev.root();
    t_node node_rev_ctx = idx.m_cst_rev.root();
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end-1) == 77777); // TODO: use UNK_SYM
    int d = 0;
    uint64_t char_pos = 0, char_pos_ctx = 0;

    //std::cout << "PKN: pattern:";
    //for (auto it = pattern_begin; it < pattern_end; ++it)
        //std::cout << " " << *it;
    //std::cout << "\n";

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end-i;
    
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM)) {
            auto timer = lm_bench::bench(timer_type::highestorder);
            // Top-level which uses actual counts rather than continuation
            // counts as in the subsequent versions. Applied to ngrams of
            // maximum length, or to ngrams starting with <s>.
            uint64_t c = 0;
            if (forward_search_wrapper(idx.m_cst_rev, node_rev, d, *start, char_pos) > 0) 
                c = idx.m_cst_rev.size(node_rev);

            // compute discount, numerator
            double D = 0;
            if (i == ngramsize)
                D = idx.discount(ngramsize);
            else // which is the special case of n<ngramsize that starts with <s>
                D = idx.discount(i, true);
            double numerator = (!unk && c - D > 0) ? (c - D) : 0;

            uint64_t lb = idx.m_cst.lb(node), rb = idx.m_cst.rb(node);
            if (backward_search_wrapper(idx.m_cst.csa, lb, rb, *start, lb, rb) > 0) {
                node = idx.m_cst.node(lb, rb);
                auto denominator = idx.m_cst.size(node);
                double N1plus_front = idx.N1PlusFront(node, start, pattern_end - 1);
                probability = (numerator / denominator) + (D * N1plus_front / denominator) * probability;
            } else {
                // TODO: check what happens here; just use backoff probability I guess
            }
        } else if (i < ngramsize && i != 1) {
            auto timer = lm_bench::bench(timer_type::lowerorder);
            // Mid-level for 2 ... n-1 grams which uses continuation counts in 
            // the KN scoring formala.
            uint64_t c = 0;
            if (forward_search_wrapper(idx.m_cst_rev, node_rev, d, *start, char_pos) > 0) 
                c = idx.N1PlusBack(node_rev, start, pattern_end);
            
            // update the context-only node in the reverse tree
            forward_search_wrapper(idx.m_cst_rev, node_rev_ctx, d-1, *start, char_pos_ctx);

            // compute discount
            double D = idx.discount(i, true);
            double numerator = (!unk && c - D > 0) ? (c - D) : 0;

            // compute N1+ components
            uint64_t lb = idx.m_cst.lb(node), rb = idx.m_cst.rb(node);
            if (backward_search_wrapper(idx.m_cst.csa, lb, rb, *start, lb, rb) > 0) { 
                node = idx.m_cst.node(lb, rb);
                auto N1plus_front = idx.N1PlusFront(node, start, pattern_end - 1);
                auto back_N1plus_front = idx.N1PlusFrontBack(node, node_rev_ctx, start, pattern_end - 1);
                d++;
                probability = (numerator / back_N1plus_front) + (D * N1plus_front / back_N1plus_front) * probability;
            } else {
                // TODO CHECK: what happens to the bounds when this is false
                node = idx.m_cst.node(lb, rb);
            }
        } else if (i == 1 || ngramsize == 1) {
            auto timer = lm_bench::bench(timer_type::lowestorder);
            // Lowest-level for 1 grams which uses continuation counts, with some
            // precomputed values as special cases to stop the iteration.
            double numerator;
            if (!unk) {
                t_pat_iter start = pattern_end-1;
                forward_search_wrapper(idx.m_cst_rev, node_rev, i-1, *start, char_pos);
                d++;
                numerator = idx.N1PlusBack(node_rev, start, pattern_end); 
            } else {
                // TODO: will the node_rev be invalid? shouldn't we still do forward_search?
                // seems values are ignored all the way up
                numerator = idx.discount(1, true);
            }
            probability = numerator / idx.m_precomputed.N1plus_dotdot;

        } else {
            assert(false);
        }
    }

    //std::cout << "PKN: returning " << probability << "\n";
    return probability;
}

// Returns the Kneser-Ney probability of the n-gram defined
// by [pattern_begin, pattern_end) where the last value is being
// predicted given the previous values in the pattern.
// Uses only a forward CST and backward search.
template <class t_idx, class t_pat_iter>
double prob_kneser_ney_forward(const t_idx& idx, 
        t_pat_iter pattern_begin, t_pat_iter pattern_end, uint64_t ngramsize)
{
    typedef typename t_idx::cst_type::node_type t_node;
    double probability = 1.0;
    t_node node_incl = idx.m_cst.root(); // matching the full pattern, including last item
    t_node node_excl = idx.m_cst.root(); // matching only the context, excluding last item
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end-1) == 77777); // TODO: use UNK_SYM

    //std::cout << "PKN: pattern:";
    //for (auto it = pattern_begin; it < pattern_end; ++it)
        //std::cout << " " << *it;
    //std::cout << "\n";

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end-i;
    
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM)) {
            auto timer = lm_bench::bench(timer_type::highestorder);
            // Top-level which uses actual counts rather than continuation
            // counts as in the subsequent versions. Applied to ngrams of
            // maximum length, or to ngrams starting with <s>.
            uint64_t c = 0;
            if (backward_search_wrapper(idx.m_cst, node_incl, *start) > 0) 
                c = idx.m_cst.size(node_incl);

            // compute discount, numerator
            double D = 0;
            if (i == ngramsize)
                D = idx.discount(ngramsize);
            else // which is the special case of n<ngramsize that starts with <s>
                D = idx.discount(i, true);
            double numerator = (!unk && c - D > 0) ? (c - D) : 0;

            if (backward_search_wrapper(idx.m_cst, node_excl, *start) > 0) {
                auto denominator = idx.m_cst.size(node_excl);
                double N1plus_front = idx.N1PlusFront(node_excl, start, pattern_end - 1);
                probability = (numerator / denominator) + (D * N1plus_front / denominator) * probability;
            } else {
                // TODO: check what happens here; just use backoff probability I guess
            }
        } else if (i < ngramsize && i != 1) {
            auto timer = lm_bench::bench(timer_type::lowerorder);
            // Mid-level for 2 ... n-1 grams which uses continuation counts in 
            // the KN scoring formala.
            uint64_t c = 0;
            if (backward_search_wrapper(idx.m_cst, node_incl, *start) > 0) 
                c = idx.N1PlusBack_from_forward(node_incl, start, pattern_end);
            
            // compute discount
            double D = idx.discount(i, true);
            double numerator = (!unk && c - D > 0) ? (c - D) : 0;

            // compute N1+ components
            if (backward_search_wrapper(idx.m_cst, node_excl, *start) > 0) { 
                auto N1plus_front = idx.N1PlusFront(node_excl, start, pattern_end - 1);
                auto back_N1plus_front = idx.N1PlusFrontBack_from_forward(node_excl, start, pattern_end - 1);
                probability = (numerator / back_N1plus_front) + (D * N1plus_front / back_N1plus_front) * probability;
            } else {
                // TODO CHECK: what happens to the bounds when this is false
            }
        } else if (i == 1 || ngramsize == 1) {
            auto timer = lm_bench::bench(timer_type::lowestorder);
            // Lowest-level for 1 grams which uses continuation counts, with some
            // precomputed values as special cases to stop the iteration.
            double numerator;
            if (!unk) {
                t_pat_iter start = pattern_end-1;
                backward_search_wrapper(idx.m_cst, node_incl, *start);
                numerator = idx.N1PlusBack_from_forward(node_incl, start, pattern_end); 
            } else {
                // TODO: will the node_incl be invalid? shouldn't we still do forward_search?
                // seems values are ignored all the way up
                numerator = idx.discount(1, true);
            }
            probability = numerator / idx.m_precomputed.N1plus_dotdot;

        } else {
            assert(false);
        }
    }

    //std::cout << "PKN: returning " << probability << "\n";
    return probability;
}
