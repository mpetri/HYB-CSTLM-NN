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

// Returns the Kneser-Ney probability of the n-gram defined
// by [pattern_begin, pattern_end) where the last value is being
// predicted given the previous values in the pattern.
// Uses forward and reversed CST.
template <class t_idx, class t_pat_iter>
double prob_mod_kneser_ney_dual(const t_idx& idx, 
        t_pat_iter pattern_begin, t_pat_iter pattern_end, uint64_t ngramsize)
{
    typedef typename t_idx::cst_type::node_type t_node;
    t_node node = idx.m_cst.root(); // v_F
    t_node node_rev_ctx = idx.m_cst_rev.root(); // v_R
    t_node node_rev = idx.m_cst_rev.root(); // v_R^all
    double p = 1.0 / idx.m_vocab.size(); // p -- FIXME: should we subtract away sentinels?

    // FIXME: there's a bug somewhere in here, as it fails the unit test (1.Perplexity)

    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end-1) == UNKNOWN_SYM); 
    uint64_t char_pos = 0, char_pos_ctx = 0;
    bool ok = !unk;

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end-i;
        if (i > 1 && *start == UNKNOWN_SYM) 
            break;

        // update match for full pattern
        if (ok) {
            ok = forward_search_wrapper(idx.m_cst_rev, node_rev, i, *start, char_pos);
        }
        // update match for context (pattern without last token)
        if (i >= 2) {
            if (backward_search_wrapper(idx.m_cst, node, *start) <= 0)
                break; // failed match means we finish
            if (i < ngramsize) 
                forward_search_wrapper(idx.m_cst_rev, node_rev_ctx, i-1, *start, char_pos_ctx);
        }

        // compute the numerator and denominator
        double D1, D2, D3p;
        idx.mkn_discount(i, D1, D2, D3p, i == 1 || i != ngramsize);

        double c, d;
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM) ) {
            c = (ok) ? idx.m_cst_rev.size(node_rev) : 0;
            d = idx.m_cst.size(node);
        } else if (i == 1 || ngramsize == 1) {
            c = (ok) ? idx.N1PlusBack(node_rev, start, pattern_end) : 0;
            d = idx.m_precomputed.N1plus_dotdot;
        } else {
            c = (ok) ? idx.N1PlusBack(node_rev, start, pattern_end) : 0;
            d = idx.N1PlusFrontBack(node, node_rev_ctx, start, pattern_end - 1); 
        }

        // update the running probability
        if (c == 1) {
            c -= D1;
        } else if (c == 2) {
            c -= D2; 
        } else if (c >= 3) {
            c -= D3p;
        }
            
        uint64_t n1, n2, n3p;
         // if it's the unigram level, the gamma can be computed using
         // precomputed quantities N1(.), N2(.), N3+(.)
	 if (i == 1 || ngramsize == 1) {
            n1 = idx.m_precomputed.N1_dot;
            n2 = idx.m_precomputed.N2_dot;
            n3p = idx.m_precomputed.N3plus_dot;
        }else{
            idx.N123PlusFront(node, start, pattern_end - 1, n1, n2, n3p);
        }

//        idx.N123PlusFront(node, start, pattern_end - 1, n1, n2, n3p);
        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;

        p = (c + gamma * p) / d;
    }

    return p;
}

// Returns the Kneser-Ney probability of the n-gram defined
// by [pattern_begin, pattern_end) where the last value is being
// predicted given the previous values in the pattern.
// Uses only a forward CST and backward search.
template <class t_idx, class t_pat_iter>
double prob_mod_kneser_ney_single(const t_idx& idx, 
        t_pat_iter pattern_begin, t_pat_iter pattern_end, uint64_t ngramsize)
{
    typedef typename t_idx::cst_type::node_type t_node;
    double p = 1.0 / idx.m_vocab.size(); // p -- FIXME: should we subtract away sentinels?
    t_node node_incl = idx.m_cst.root(); // v_F^all matching the full pattern, including last item
    t_node node_excl = idx.m_cst.root(); // v_F     matching only the context, excluding last item
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end-1) == UNKNOWN_SYM);
    bool ok = !unk;

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end-i;
        if (i > 1 && *start == UNKNOWN_SYM) 
            break;

        // update the two searches into the CST
        if (ok) {
            ok = backward_search_wrapper(idx.m_cst, node_incl, *start);
        }
        if (i >= 2) {
            if (backward_search_wrapper(idx.m_cst, node_excl, *start) <= 0)
                break;
        }

        // compute the count and normaliser
        double D1, D2, D3p;
        idx.mkn_discount(i, D1, D2, D3p, i == 1 || i != ngramsize);

        double c, d;
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM) ) {
            c = (ok) ? idx.m_cst.size(node_incl) : 0;
            d = idx.m_cst.size(node_excl);
        } else if (i == 1 || ngramsize == 1) {
            c = (ok) ? idx.N1PlusBack_from_forward(node_incl, start, pattern_end) : 0;
            d = idx.m_precomputed.N1plus_dotdot;
        } else {
            c = (ok) ? idx.N1PlusBack_from_forward(node_incl, start, pattern_end) : 0;
            d = idx.N1PlusFrontBack_from_forward(node_excl, start, pattern_end - 1);
        }

        // update the running probability
        if (c == 1) {
            c -= D1;
        } else if (c == 2) {
            c -= D2; 
        } else if (c >= 3) {
            c -= D3p;
        }
            
        uint64_t n1, n2, n3p;
        idx.N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p);
        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;
        p = (c + gamma * p) / d;
    }

    return p;
}
