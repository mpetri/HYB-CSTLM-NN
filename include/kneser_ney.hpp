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

template <class t_idx, class t_atom>
struct LMQueryMKN
{
    LMQueryMKN(t_idx *idx, int ngramsize);
    // need a copy / assignment constructor 
    // and a serialisation of the memory as state definition

    double append_symbol(const t_atom &symbol);

    int m_ngramsize;
    t_idx *m_idx;
    typedef typename t_idx::cst_type::node_type t_node;
    std::vector<t_node> m_last_nodes_incl;
    std::deque<t_atom> m_pattern;
};

template <class t_idx, class t_atom>
LMQueryMKN::LMQueryMKN(t_idx *idx, int ngramsize)
    : m_idx(idx), m_ngramsize(ngramsize)
{
    m_last_nodes_incl.push_back(m_idx->m_cst.root());
    m_pattern.push_back(PAT_START_SYM);
}

template <class t_idx, class t_atom>
double LMQueryMKN::append_symbol(const t_atom &symbol)
{
    m_pattern.push_back(symbol);
    while (ngramsize > 0 && m_pattern.size() > ngramsize) 
        m_pattern.pop_front();
#ifdef STATELESS_QUERY
    // slow way
    return prob_mod_kneser_ney_single(m_idx, m_pattern.begin(), m_pattern.end(), m_ngramsize);
#else
    // fast way, tracking state
    typedef typename t_idx::cst_type::node_type t_node;
    double p = 1.0 / (idx.m_vocab.size()-4);        // p -- FIXME: should we subtract away sentinels? //ehsan: not sure why -4 works! but it works!
    t_node node_incl = idx.m_cst.root();            // v_F^all matching the full pattern, including last item
    auto node_excl_it = m_last_nodes_incl.begin();  // v_F     matching only the context, excluding last item
    t_node node_excl = *node_excl_it;
    auto pattern_begin = m_pattern.begin();
    auto pattern_end = m_pattern.end();
    
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end-1) == UNKNOWN_SYM);
    bool ok = !unk;
    std::vector<t_node> node_incl_vec({ node_incl });

    for (unsigned i = 1; i <= size; ++i) {
        t_pat_iter start = pattern_end-i;
        if (i > 1 && *start == UNKNOWN_SYM) 
            break;

        //LOG(INFO) << "pattern is: " << idx.m_vocab.id2token(start, pattern_end);

        // update the two searches into the CST
        if (ok) {
            ok = backward_search_wrapper(idx.m_cst, node_incl, *start);
            //LOG(INFO) << "\tpattern lookup, ok=" << ok << " node=" << node_incl;
            if (ok) node_incl_vec.push_back(node_incl);
        }
        if (i >= 2) {
            node_excl_it++;
            //LOG(INFO) << "context query for: " << idx.m_vocab.id2token(start, pattern_end-1);
            if (node_excl_it == m_last_nodes_incl.end()) {
                //LOG(INFO) << "\tfailed context lookup; quitting";
                break;
            } else {
                node_excl = *node_excl_it;
            }
        }

        // compute the count and normaliser
        double D1, D2, D3p;
        //LOG(INFO) << "test for continuation counts: " << (i == 1 || i != ngramsize) << " i: " << i << " ngramsize: " << ngramsize;
        idx.mkn_discount(i, D1, D2, D3p, i == 1 || i != ngramsize);

        double c, d;
        uint64_t n1 = 0, n2 = 0, n3p = 0;
        if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM) ) {
            c = (ok) ? idx.m_cst.size(node_incl) : 0;
            d = idx.m_cst.size(node_excl);
            idx.N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p); // does this work for node_excl = root?
            //LOG(INFO) << "highest level c=" << c << " d=" << d << " n1=" << n1 << " n2=" << n2 << " n3p=" << n3p;
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
            //LOG(INFO) << "mid/low level c=" << c << " d=" << d << " n1=" << n1 << " n2=" << n2 << " n3p=" << n3p;
        }

        // update the running probability
        if (c == 1) { c -= D1; } 
        else if (c == 2) { c -= D2; } 
        else if (c >= 3) { c -= D3p; }

        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;
        p = (c + gamma * p) / d;
        //LOG(INFO) << "adjusted c=" << c << " gamma=" << gamma << " gamma/d=" << (gamma/d) << " p=" << p << " log(p)=" << log10(p);
        //LOG(INFO) << "\tdiscounts: D1=" << D1 << " D2=" << D2 << " D3p=" << D3p;
    }

    //LOG(INFO) << "FINAL prob " << p;
    // update the state for the next call 
    m_last_nodes_incl = node_incl_vec;
    while (m_pattern.size() > m_last_nodes_incl)
        m_pattern.pop_front();

    return p;
#endif
}
