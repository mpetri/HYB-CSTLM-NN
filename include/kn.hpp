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
double prob_kneser_ney(const t_idx& idx,
                       t_pat_iter pattern_begin, t_pat_iter pattern_end, uint64_t ngramsize)
{
    typedef typename t_idx::cst_type::node_type t_node;
    double p = 1.0;
    t_node node_incl = idx.m_cst.root(); // v_F^all matching the full pattern, including last item
    t_node node_excl = idx.m_cst.root(); // v_F     matching only the context, excluding last item
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end - 1) == UNKNOWN_SYM);
    bool ok = !unk;

    //LOG(INFO) << "prob_kneser_ney_single for pattern: " << idx.m_vocab.id2token(pattern_begin, pattern_end);
    //LOG(INFO) << "as numbers: " << std::vector<uint64_t>(pattern_begin, pattern_end);

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end - i;
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
        double D = idx.discount(i, i == 1 || i != ngramsize);
        double c, d;
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM)) {
            c = (ok) ? idx.m_cst.size(node_incl) : 0;
            d = idx.m_cst.size(node_excl);
        } else if (i == 1 || ngramsize == 1) {
            c = (ok) ? idx.N1PlusBack(node_incl, start, pattern_end) : D;
            d = idx.m_precomputed.N1plus_dotdot;
        } else {
            c = (ok) ? idx.N1PlusBack(node_incl, start, pattern_end) : 0;
            d = idx.N1PlusFrontBack(node_excl, start, pattern_end - 1);
        }

        // update the running probability
        if (i > 1) {
            double q = idx.N1PlusFront(node_excl, start, pattern_end - 1);
            p = (std::max(c - D, 0.0) + D * q * p) / d;
        } else {
            p = c / d;
        }
    }

    //LOG(INFO) << "prob_kneser_ney_single returning: " << p;

    return p;
}
