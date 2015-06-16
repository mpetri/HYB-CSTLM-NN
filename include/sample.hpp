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
#include <random>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"
#include "constants.hpp"

template <class t_idx, class t_pat_iter, class t_rng>
uint64_t sample_next_symbol(const t_idx& idx, t_pat_iter pattern_begin, t_pat_iter pattern_end, 
        uint64_t ngramsize, t_rng &rng)
{
    return _sample_next_symbol(idx, pattern_begin, pattern_end, ngramsize, idx.m_cst.root(), 0, rng);
}

template <class t_idx, class t_pat_iter, class t_rng>
uint64_t _sample_next_symbol(const t_idx& idx, 
        t_pat_iter pattern_begin, 
        t_pat_iter pattern_end, 
        uint64_t ngramsize,
        typename t_idx::cst_type::node_type node,
        uint64_t ctxsize,
        t_rng &rng)
{
    typedef typename t_idx::cst_type::node_type t_node;
    size_t size = std::distance(pattern_begin, pattern_end);

    // attempt a longer match
    uint64_t next = EOF_SYM;
    if (ctxsize < size) {
        t_pat_iter start = pattern_end-(ctxsize+1);
        if (*start != UNKNOWN_SYM) {
            auto ok = backward_search_wrapper(idx.m_cst, node, *start);
            if (ok) {
                next = _sample_next_symbol(idx, pattern_begin, pattern_end, 
                        ngramsize, node, ctxsize+1, rng);
                if (next != EOF_SYM)
                    return next;
            }
        }
    } 

    // either longer match failed, or we're at maximum length
    auto i = ctxsize + 1;
    double D = idx.discount(i, i == 1 || i != ngramsize);
    double d;
    auto start = (pattern_end-i);
    if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM) ) {
        d = idx.m_cst.size(node);
    } else if (i == 1 || ngramsize == 1) {
        d = idx.m_precomputed.N1plus_dotdot;
    } else {
        d = idx.N1PlusFrontBack_from_forward(node, start, pattern_end);
    }

    if (i > 1) {
        double q = idx.N1PlusFront(node, start, pattern_end - 1);
        double stay = d / D * q - 1;

        std::uniform_real_distribution<> uniform(0, 1);
        double r = uniform(rng);
        if (r < stay) {
            uint64_t r = uniform(rng) * d;
            // read off the symbol from the corresponding edge
            auto child = idx.m_cst.child(node, 1);
            while (child != idx.m_cst.root())
            {
                r -= idx.m_cst.size() - D;
                if (r <= 0) {
                    next = idx.m_cst.edge(child, i);
                    break;
                }
                child = idx.m_cst.sibling(child);
            }
            assert(false && "you shouldn't reach this line");
        } else {
            // backoff
            next = EOF_SYM;
        }
    } else {
        // FIXME: somewhat wasteful, could be done in one iterator
        static std::vector<uint64_t> unigrams_cs(unigram_counts(idx));
        static std::discrete_distribution<uint64_t> unigrams(unigrams_cs.begin(), unigrams_cs.end());
        unigrams_cs.clear();
        next = unigrams(rng);
    }

    return next;
}

template <class t_idx>
std::vector<uint64_t> unigram_counts(const t_idx &idx) 
{
    // FIXME: this should be precomputed; and perhaps we can do this faster?
    auto root = idx.m_cst.root();
    std::vector<uint64_t> pattern(1, NUM_SPECIAL_SYMS); // place-holder pattern
    std::vector<uint64_t> weights;
    for (const auto& child : idx.m_cst.children(root))
        weights.push_back(idx.N1PlusBack_from_forward(child, pattern.begin(), pattern.end()));
    return weights;
}
