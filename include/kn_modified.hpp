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
    double p = 1.0 / (idx.m_vocab.size()-4); // p -- FIXME: should we subtract away sentinels? //ehsan: not sure why -4 works! but it works!

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
        //if it's the unigram level, the gamma can be computed using
        // n1_cnt, n2_cnt, vocab_size
        // have a look at ModKneserNey::lowerOrderWeight function of srilm in
        // Discount.cc 
        if (i == 1 || ngramsize == 1) {
            n1 = idx.m_precomputed.n1_cnt[1];
            n2 = idx.m_precomputed.n2_cnt[1];
            n3p = (idx.vocab_size()-2)-(n1 + n2);
        } else {
            // FIXME: need N1PlusFrontBack for middle orders
            idx.N123PlusFront(node, start, pattern_end - 1, n1, n2, n3p);
        }

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
    double p = 1.0 / (idx.m_vocab.size()-4); // p -- FIXME: should we subtract away sentinels? //ehsan: not sure why -4 works! but it works!
    t_node node_incl = idx.m_cst.root(); // v_F^all matching the full pattern, including last item
    t_node node_excl = idx.m_cst.root(); // v_F     matching only the context, excluding last item
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end-1) == UNKNOWN_SYM);
    bool ok = !unk;

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end-i;
        if (i > 1 && *start == UNKNOWN_SYM) 
            break;

        LOG(INFO) << "pattern is: " << idx.m_vocab.id2token(start, pattern_end);

        // update the two searches into the CST
        if (ok) {
        //    LOG(INFO)<<"**start is: "<<idx.m_vocab.id2token(*start)<<endl;
            ok = backward_search_wrapper(idx.m_cst, node_incl, *start);
            LOG(INFO) << "\tpattern lookup, ok=" << ok;
        }
        if (i >= 2) {
        //    LOG(INFO)<<"*start is: "<<idx.m_vocab.id2token(*start)<<endl;
            if (backward_search_wrapper(idx.m_cst, node_excl, *start) <= 0) {
                LOG(INFO) << "\tfailed context lookup; quitting";
                break;
            }
        }

        // compute the count and normaliser
        double D1, D2, D3p;
        LOG(INFO) << "test for continuation counts: " << (i == 1 || i != ngramsize) << " i: " << i << " ngramsize: " << ngramsize;
        idx.mkn_discount(i, D1, D2, D3p, i == 1 || i != ngramsize);

        double c, d;
        uint64_t n1 = 0, n2 = 0, n3p = 0;
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM) ) {
            c = (ok) ? idx.m_cst.size(node_incl) : 0;
            d = idx.m_cst.size(node_excl);
            idx.N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p); // does this work for node_excl = root?
            LOG(INFO) << "highest level c=" << c << " d=" << d << " n1=" << n1 << " n2=" << n2 << " n3p=" << n3p;
        } else {
            c = (ok) ? idx.N1PlusBack_from_forward(node_incl, start, pattern_end) : 0;
            if (i == 1 || ngramsize == 1) {
                // lowest level
                d = idx.m_precomputed.N1plus_dotdot;
                n1 = idx.m_precomputed.n1_cnt[1];
                n2 = idx.m_precomputed.n2_cnt[1];
                n3p = idx.m_precomputed.N3plus_dot;
            } else {
                // mid level (most cases arrive here)
                d = idx.N1PlusFrontBack_from_forward(node_excl, start, pattern_end - 1); // is this right?
                idx.N123PlusFrontBack_from_forward(node_excl, start, pattern_end - 1, n1, n2, n3p);
            }
            LOG(INFO) << "mid/low level c=" << c << " d=" << d << " n1=" << n1 << " n2=" << n2 << " n3p=" << n3p;
        }

        // update the running probability
        if (c == 1) { c -= D1; } 
        else if (c == 2) { c -= D2; } 
        else if (c >= 3) { c -= D3p; }

        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;
        p = (c + gamma * p) / d;
        LOG(INFO) << "adjusted c=" << c << " gamma=" << gamma << " gamma/d=" << (gamma/d) << " p=" << p << " log(p)=" << log10(p);
        LOG(INFO) << "\tdiscounts: D1=" << D1 << " D2=" << D2 << " D3p=" << D3p;
        //LOG(INFO)<<"n1 = "<<n1<<" n2 = "<<n2<<" n3p = "<<n3p<<endl;
	//LOG(INFO)<<"D1 = "<<D1<<" D2 = "<<D2<<" D3p = "<<D3p<<endl;
        //LOG(INFO)<<"gamma = "<<gamma/d<<" log10(gamma)= "<<log10(gamma/d)<<endl;
	//LOG(INFO) << "pattern is: " << idx.m_vocab.id2token(pattern_begin, pattern_end);
        //LOG(INFO)<<"Pattern is: "<<std::vector<u_int64_t>(pattern_begin,pattern_end);
        //LOG(INFO)<<"probability is: "<<p<<" log10(probability) is: "<<log10(p)<<endl;
        //LOG(INFO)<<"----------------------------------"<<endl;
    }

    LOG(INFO) << "FINAL prob " << p;

    return p;
}
