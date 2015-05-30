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
double sentence_logprob_kneser_ney(const t_idx& idx, const t_pattern& word_vec, uint64_t& M, uint64_t ngramsize, bool fast_index)
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
/*
        if (pattern.back() == UNKNOWN_SYM) {
            M = M - 1; // excluding OOV from perplexity - identical to SRILM ppl
        }
*/
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
double sentence_perplexity_kneser_ney(const t_idx& idx, t_pattern &pattern, uint32_t ngramsize, bool fast_index)
{
    auto pattern_size = pattern.size();
    pattern.push_back(PAT_END_SYM);
    pattern.insert(pattern.begin(), PAT_START_SYM);
    // run the query
    uint64_t M = pattern_size + 1;
    double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize, fast_index);
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
    bool unk = (*(pattern_end-1) == UNKNOWN_SYM); 
    int d = 0;
    uint64_t char_pos = 0, char_pos_ctx = 0;

    //std::cout << "PKN: pattern:";
    //for (auto it = pattern_begin; it < pattern_end; ++it)
        //std::cout << " " << *it;
    //std::cout << "\n";
    //LOG(INFO) << std::vector<uint64_t>(pattern_begin, pattern_end);

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end-i;
        if (i > 1 && *start == UNKNOWN_SYM) 
            break;
    
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM)) {
            auto timer = lm_bench::bench(timer_type::highestorder);
            // Top-level which uses actual counts rather than continuation
            // counts as in the subsequent versions. Applied to ngrams of
            // maximum length, or to ngrams starting with <s>.
            uint64_t c = 0;
            if (!unk && forward_search_wrapper(idx.m_cst_rev, node_rev, d, *start, char_pos) > 0) 
                c = idx.m_cst_rev.size(node_rev);

            // compute discount, numerator
            double D = 0;
            if (i == ngramsize)
                D = idx.discount(ngramsize);
            else // which is the special case of n<ngramsize that starts with <s>
                D = idx.discount(i, true);
            double numerator = (!unk && c - D > 0) ? (c - D) : 0;
            //std::cout << "\tsize " << i << " (top): numer=" << numerator << "\n";

            uint64_t lb = idx.m_cst.lb(node), rb = idx.m_cst.rb(node);
            if (backward_search_wrapper(idx.m_cst.csa, lb, rb, *start, lb, rb) > 0) {
                node = idx.m_cst.node(lb, rb);
                auto denominator = idx.m_cst.size(node);
                double N1plus_front = idx.N1PlusFront(node, start, pattern_end - 1);
                probability = (numerator / denominator) + (D * N1plus_front / denominator) * probability;
                //std::cout << "\tsize " << i << " (top): " << probability << "\n";
                //std::cout << "\tsize " << i << " (top): N1+f=" << N1plus_front << " D=" << D << " denom=" << denominator << "\n";
            } else {
                // retain backoff probability
                //std::cout << "\tsize " << i << " (top): fall-through\n";
            }
        } else if (i < ngramsize && i != 1) {
            auto timer = lm_bench::bench(timer_type::lowerorder);
            // Mid-level for 2 ... n-1 grams which uses continuation counts in 
            // the KN scoring formala.
            uint64_t c = 0;
            if (!unk && forward_search_wrapper(idx.m_cst_rev, node_rev, d, *start, char_pos) > 0) 
                c = idx.N1PlusBack(node_rev, start, pattern_end);
            
            // update the context-only node in the reverse tree
            forward_search_wrapper(idx.m_cst_rev, node_rev_ctx, d-1, *start, char_pos_ctx);

            // compute discount
            double D = idx.discount(i, true);
            double numerator = (!unk && c - D > 0) ? (c - D) : 0;
            //std::cout << "\tsize " << i << " (mid): numer=" << numerator << "\n";

            // compute N1+ components
            uint64_t lb = idx.m_cst.lb(node), rb = idx.m_cst.rb(node);
            if (backward_search_wrapper(idx.m_cst.csa, lb, rb, *start, lb, rb) > 0) { 
                node = idx.m_cst.node(lb, rb);
                auto N1plus_front = idx.N1PlusFront(node, start, pattern_end - 1);
                auto back_N1plus_front = idx.N1PlusFrontBack(node, node_rev_ctx, start, pattern_end - 1); // bug in here somewhere
                d++;
                probability = (numerator / back_N1plus_front) + (D * N1plus_front / back_N1plus_front) * probability;
                //std::cout << "\tsize " << i << " (mid): " << probability << "\n";
                //std::cout << "\tsize " << i << " (mid): N1+f=" << N1plus_front << " D=" << D << " denom=" << back_N1plus_front << "\n";
            } else {
                // retain backoff probability
                //std::cout << "\tsize " << i << " (mid): fall-through\n";
                break;
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
                //std::cout << "\t\tunigram seen " << numerator << " node: [" << idx.m_cst_rev.lb(node_rev) << ", " << idx.m_cst_rev.rb(node_rev) << "]" << std::endl;
            } else {
                numerator = idx.discount(1, true);
                //std::cout << "\t\tunigram unk " << numerator << std::endl;
            }
            probability = numerator / idx.m_precomputed.N1plus_dotdot;
            //std::cout << "\tsize 1: " << probability << "\n";

        } else {
            assert(false);
        }
    }

    //std::cout << "PKN: returning " << probability << "\n";
    return probability;
}

extern std::vector<uint32_t> ngram_occurrences;

// Returns the Kneser-Ney probability of the n-gram defined
// by [pattern_begin, pattern_end) where the last value is being
// predicted given the previous values in the pattern.
// Uses only a forward CST and backward search.
template <class t_idx, class t_pat_iter>
double prob_kneser_ney_forward(const t_idx& idx, 
        t_pat_iter pattern_begin, t_pat_iter pattern_end, uint64_t ngramsize,
        int cache_limit=0)
{
    typedef typename t_idx::cst_type::node_type t_node;
    double probability = 1.0;
    t_node node_incl = idx.m_cst.root(); // matching the full pattern, including last item
    t_node node_excl = idx.m_cst.root(); // matching only the context, excluding last item
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end-1) == UNKNOWN_SYM);

    // simple cache which saves all expensive calls bar a single 'backward_search' each iteration
    typedef typename t_idx::cst_type::size_type size_type;
    typedef std::pair<double, t_node> cache_type;
    static std::unordered_map<size_type, cache_type> ngram_cache;
    static uint64_t cache_hits = 0, cache_misses = 0;

    LOG_EVERY_N(1000, INFO) << "PKN cache stats: " << cache_hits << " hits and " << cache_misses << " misses";

    //LOG(INFO) << "PKN: pattern = " << std::vector<uint64_t>(pattern_begin, pattern_end);

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end-i;
        if (i > 1 && *start == UNKNOWN_SYM) 
            break;

        bool incl_pattern_found = false;
        if (!unk && backward_search_wrapper(idx.m_cst, node_incl, *start) > 0) {
            incl_pattern_found = true;
            // check cache 
            if (i <= cache_limit) {
                auto it = ngram_cache.find(idx.m_cst.id(node_incl));
                if (it != ngram_cache.end()) {
                    probability = it->second.first;
                    node_excl = it->second.second;
                    cache_hits += 1;
                    continue;
                } else {
                    cache_misses += 1;
                }
            }
        }
    
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM)) {
            auto timer = lm_bench::bench(timer_type::highestorder);
            // Top-level which uses actual counts rather than continuation
            // counts as in the subsequent versions. Applied to ngrams of
            // maximum length, or to ngrams starting with <s>.
            uint64_t c = 0;
            if (incl_pattern_found)
                c = idx.m_cst.size(node_incl);

            // compute discount, numerator
            double D = 0;
            if (i == ngramsize)
                D = idx.discount(ngramsize);
            else // which is the special case of n<ngramsize that starts with <s>
                D = idx.discount(i, true);
            double numerator = (!unk && c - D > 0) ? (c - D) : 0;
            //LOG(INFO) << "\tsize " << i << " (top): numer=" << numerator;

            if (backward_search_wrapper(idx.m_cst, node_excl, *start) > 0) {
                auto denominator = idx.m_cst.size(node_excl);
                double N1plus_front = idx.N1PlusFront(node_excl, start, pattern_end - 1);
                probability = (numerator / denominator) + (D * N1plus_front / denominator) * probability;
                //LOG(INFO) << "\tsize " << i << " (top): " << probability;
                //LOG(INFO) << "\tsize " << i << " (top): N1+f=" << N1plus_front << " D=" << D << " denom=" << denominator;
                if (ngram_occurrences.size() <= i)
                    ngram_occurrences.resize(i+1);
                ngram_occurrences[i] += 1;
            } else {
                // just use backoff probability 
                //LOG(INFO) << "\tsize " << i << " (top): fall-through";
            }
        } else if (i < ngramsize && i != 1) {
            auto timer = lm_bench::bench(timer_type::lowerorder);
            // Mid-level for 2 ... n-1 grams which uses continuation counts in 
            // the KN scoring formala.
            uint64_t c = 0;
            if (incl_pattern_found)
                c = idx.N1PlusBack_from_forward(node_incl, start, pattern_end);
            
            // compute discount
            double D = idx.discount(i, true);
            double numerator = (!unk && c - D > 0) ? (c - D) : 0;
            //LOG(INFO) << "\tsize " << i << " (mid): numer=" << numerator << "\n";

            // compute N1+ components
            if (backward_search_wrapper(idx.m_cst, node_excl, *start) > 0) { 
                auto N1plus_front = idx.N1PlusFront(node_excl, start, pattern_end - 1);
                auto back_N1plus_front = idx.N1PlusFrontBack_from_forward(node_excl, start, pattern_end - 1);
                probability = (numerator / back_N1plus_front) + (D * N1plus_front / back_N1plus_front) * probability;
                //LOG(INFO) << "\tsize " << i << " (mid): " << probability;
                //LOG(INFO) << "\tsize " << i << " (mid): N1+f=" << N1plus_front << " D=" << D << " denom=" << back_N1plus_front;
                if (ngram_occurrences.size() <= i)
                    ngram_occurrences.resize(i+1);
                ngram_occurrences[i] += 1;
            } else {
                // just use backoff probability 
                //LOG(INFO) << "\tsize " << i << " (mid): fall-through";
                break;
            }
        } else if (i == 1 || ngramsize == 1) {
            auto timer = lm_bench::bench(timer_type::lowestorder);
            // Lowest-level for 1 grams which uses continuation counts, with some
            // precomputed values as special cases to stop the iteration.
            double numerator;
            if (!unk) {
                t_pat_iter start = pattern_end-1;
                assert(incl_pattern_found);
                numerator = idx.N1PlusBack_from_forward(node_incl, start, pattern_end); 
                //LOG(INFO) << "\t\tunigram, not UNK numer: " << numerator << " node: [" << idx.m_cst.lb(node_incl) << ", " << idx.m_cst.rb(node_incl) << "]";
                if (ngram_occurrences.size() <= i)
                    ngram_occurrences.resize(i+1);
                ngram_occurrences[i] += 1;
            } else {
                // TODO: will the node_incl be invalid? shouldn't we still do forward_search?
                // seems values are ignored all the way up
                numerator = idx.discount(1, true);
                //LOG(INFO) << "\t\tunigram, UNK numer: " << numerator;
            }
            probability = numerator / idx.m_precomputed.N1plus_dotdot;
            //LOG(INFO) << "\tsize 1: " << probability;
        } else {
            assert(false);
        }

        // update the cache
        if (!unk && i <= cache_limit) {
            ngram_cache[idx.m_cst.id(node_incl)] = std::make_pair(probability, node_excl);
        }
    }

    //LOG(INFO) << "PKN: returning " << probability;
    return probability;
}
