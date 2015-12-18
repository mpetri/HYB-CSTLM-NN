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
// Uses only a forward CST and backward search.
template <class t_idx, class t_pat_iter>
double prob_mod_kneser_ney(const t_idx& idx, t_pat_iter pattern_begin,
                           t_pat_iter pattern_end, uint64_t ngramsize,
                           bool isfishy)
{
    typedef typename t_idx::cst_type::node_type t_node;
    double p = 1.0 / (idx.m_vocab.size() - 4); // p -- FIXME: should we subtract
    // away sentinels? //ehsan: not
    // sure why -4 works! but it works!
    t_node node_incl = idx.m_cst
                           .root(); // v_F^all matching the full pattern, including last item
    t_node node_excl = idx.m_cst
                           .root(); // v_F     matching only the context, excluding last item
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end - 1) == UNKNOWN_SYM);
    bool ok = !unk;

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end - i;
        if (i > 1 && *start == UNKNOWN_SYM)
            break;

        // update the two searches into the CST
        if (ok) {
            //    LOG(INFO)<<"**start is: "<<idx.m_vocab.id2token(*start)<<endl;
            ok = backward_search_wrapper(idx.m_cst, node_incl, *start);
        }
        if (i >= 2) {
            //    LOG(INFO)<<"*start is: "<<idx.m_vocab.id2token(*start)<<endl;
            if (backward_search_wrapper(idx.m_cst, node_excl, *start) <= 0)
                break;
        }

        // compute the count and normaliser
        double D1, D2, D3p;
        idx.mkn_discount(i, D1, D2, D3p, i == 1 || i != ngramsize);

        double c, d;
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM)) {
            c = (ok) ? idx.m_cst.size(node_incl) : 0;
            // LOG(INFO)<<"idx.m_cst.size(node_incl) is:
            // "<<idx.m_cst.size(node_incl)<<endl;
            d = idx.m_cst.size(node_excl);
            // LOG(INFO)<<"denominator: "<<d<<endl;
            // LOG(INFO)<<"Highest Level: "<<c<<endl;
        } else if (i == 1 || ngramsize == 1) {
            c = (ok) ? idx.N1PlusBack(node_incl, start, pattern_end) : 0;
            d = idx.m_precomputed.N1plus_dotdot;
            // LOG(INFO)<<"denominator: "<<d<<endl;
            // LOG(INFO)<<"Lowest Level: "<<c<<endl;
        } else {
            c = (ok) ? idx.N1PlusBack(node_incl, start, pattern_end) : 0;
            d = idx.N1PlusFrontBack(node_excl, start, pattern_end - 1);
            // LOG(INFO)<<"denominator: "<<d<<endl;
            // LOG(INFO)<<"Lower Level: "<<c<<endl;
        }

        // update the running probability
        if (c == 1) {
            // LOG(INFO)<<"D1 is: "<<D1<<endl;
            c -= D1;
        } else if (c == 2) {
            // LOG(INFO)<<"D2 is: "<<D2<<endl;
            c -= D2;
        } else if (c >= 3) {
            // LOG(INFO)<<"D3p is: "<<D3p<<endl;
            c -= D3p;
        }

        uint64_t n1, n2, n3p;
        // if it's the unigram level, the gamma can be computed using
        // n1_cnt, n2_cnt, vocab_size
        // have a look at ModKneserNey::lowerOrderWeight function of srilm in
        // Discount.cc
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM)) {
            idx.N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p);
        } else if (i == 1 || ngramsize == 1) {
            n1 = idx.m_precomputed.n1_cnt[1];
            n2 = idx.m_precomputed.n2_cnt[1];
            n3p = (idx.vocab_size() - 2) - (n1 + n2);
        } else {
            if (!isfishy)
                // idx.N123PlusFront_lower(node_excl, start, pattern_end - 1, n1, n2,
                // n3p); //accurate version
                idx.N123PlusFrontPrime(node_excl, start, pattern_end - 1, n1, n2, n3p);
            else
                idx.N123PlusFront(node_excl, start, pattern_end - 1, n1, n2,
                                  n3p); // FishyVersion}
            // idx.N123PlusFrontBack_from_forward(node_excl, start, pattern_end - 1,
            // n1, n2, n3p);//XXX Do not use this.
        }
        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;
        p = (c + gamma * p) / d;
        // LOG(INFO)<<"n1 = "<<n1<<" n2 = "<<n2<<" n3p = "<<n3p<<endl;
        // LOG(INFO)<<"D1 = "<<D1<<" D2 = "<<D2<<" D3p = "<<D3p<<endl;
        // LOG(INFO)<<"gamma = "<<gamma/d<<" log10(gamma)= "<<log10(gamma/d)<<endl;
        // LOG(INFO) << "pattern is: " << idx.m_vocab.id2token(pattern_begin,
        // pattern_end);
        // LOG(INFO)<<"Pattern is:
        // "<<std::vector<u_int64_t>(pattern_begin,pattern_end);
        // LOG(INFO)<<"probability is: "<<p<<" log10(probability) is:
        // "<<log10(p)<<endl;
        // LOG(INFO)<<"----------------------------------"<<endl;
    }

    return p;
}
