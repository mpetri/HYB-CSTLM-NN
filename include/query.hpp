#pragma once

//#define STATELESS_QUERY

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
#ifdef STATELESS_QUERY
#include "kn_modified.hpp"
#endif

// Returns the Kneser-Ney probability of a sentence, word at a
// time. Words are supplied using the append_symbol method which
// returns the conditional probability of that word given all
// previous words.

template <class t_idx, class t_atom>
class LMQueryMKN
{
public:
    LMQueryMKN(const t_idx *idx, uint64_t ngramsize);
    double append_symbol(const t_atom &symbol);
    int compare(const LMQueryMKN &other) const;

private:
    const t_idx *m_idx;
    uint64_t m_ngramsize;
    typedef typename t_idx::cst_type::node_type t_node;
    std::vector<t_node> m_last_nodes_incl;
    std::deque<t_atom> m_pattern;
};

template <class t_idx, class t_atom>
LMQueryMKN<t_idx, t_atom>::LMQueryMKN(const t_idx *idx, uint64_t ngramsize)
    : m_idx(idx), m_ngramsize(ngramsize)
{
    m_last_nodes_incl.push_back(m_idx->m_cst.root());
    m_pattern.push_back(PAT_START_SYM);
}

template <class t_idx, class t_atom>
double LMQueryMKN<t_idx, t_atom>::append_symbol(const t_atom &symbol)
{
    if (symbol == PAT_START_SYM && m_pattern.size() == 1 && m_pattern.front() == PAT_START_SYM)
        return 1;

    m_pattern.push_back(symbol);
    while (m_ngramsize > 0 && m_pattern.size() > m_ngramsize) 
        m_pattern.pop_front();
    std::vector<t_atom> pattern(m_pattern.begin(), m_pattern.end());
#ifdef STATELESS_QUERY
    // slow way
    return prob_mod_kneser_ney_single(*m_idx, pattern.begin(), pattern.end(), m_ngramsize);
#else
    // fast way, tracking state
    typedef typename t_idx::cst_type::node_type t_node;
    double p = 1.0 / (m_idx->m_vocab.size()-4);        // p -- FIXME: should we subtract away sentinels? //ehsan: not sure why -4 works! but it works!
    t_node node_incl = m_idx->m_cst.root();            // v_F^all matching the full pattern, including last item
    auto node_excl_it = m_last_nodes_incl.begin();  // v_F     matching only the context, excluding last item
    t_node node_excl = *node_excl_it;
    auto pattern_begin = pattern.begin();
    auto pattern_end = pattern.end();
    
    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end-1) == UNKNOWN_SYM);
    bool ok = !unk;
    std::vector<t_node> node_incl_vec({ node_incl });

    for (unsigned i = 1; i <= size; ++i) {
        auto start = pattern_end-i;
        if (i > 1 && *start == UNKNOWN_SYM) 
            break;

        //LOG(INFO) << "pattern is: " << m_idx->m_vocab.id2token(start, pattern_end);

        // update the two searches into the CST
        if (ok) {
            ok = backward_search_wrapper(m_idx->m_cst, node_incl, *start);
            //LOG(INFO) << "\tpattern lookup, ok=" << ok << " node=" << node_incl;
            if (ok) node_incl_vec.push_back(node_incl);
        }
        if (i >= 2) {
            node_excl_it++;
            //LOG(INFO) << "context query for: " << m_idx->m_vocab.id2token(start, pattern_end-1);
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
        m_idx->mkn_discount(i, D1, D2, D3p, i == 1 || i != m_ngramsize);

        double c, d;
        uint64_t n1 = 0, n2 = 0, n3p = 0;
        if ((i == m_ngramsize && m_ngramsize != 1) || (*start == PAT_START_SYM) ) {
            c = (ok) ? m_idx->m_cst.size(node_incl) : 0;
            d = m_idx->m_cst.size(node_excl);
            m_idx->N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p); // does this work for node_excl = root?
            //LOG(INFO) << "highest level c=" << c << " d=" << d << " n1=" << n1 << " n2=" << n2 << " n3p=" << n3p;
        } else {
            c = (ok) ? m_idx->N1PlusBack_from_forward(node_incl, start, pattern_end) : 0;
            if (i == 1 || m_ngramsize == 1) {
                // lowest level
                d = m_idx->m_precomputed.N1plus_dotdot;
                n1 = m_idx->m_precomputed.n1_cnt[1];
                n2 = m_idx->m_precomputed.n2_cnt[1];
                n3p = m_idx->m_precomputed.N3plus_dot;
            } else {
                // mid level (most cases arrive here)
                d = m_idx->N1PlusFrontBack_from_forward(node_excl, start, pattern_end - 1); // is this right?
                m_idx->N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p);
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
    while (m_pattern.size() > m_last_nodes_incl.size())
        m_pattern.pop_front();

    return p;
#endif
}

template <class t_idx, class t_atom>
int LMQueryMKN<t_idx,t_atom>::compare(const LMQueryMKN &other) const
{
    if (m_idx < other.m_idx) return -1;
    if (m_idx > other.m_idx) return +1;
    if (m_pattern.size() < other.m_pattern.size()) return -1;
    if (m_pattern.size() > other.m_pattern.size()) return +1;
    if (m_last_nodes_incl.size() < other.m_last_nodes_incl.size()) return -1;
    if (m_last_nodes_incl.size() > other.m_last_nodes_incl.size()) return +1;
    for (auto i = 0u; i < m_pattern.size(); ++i) {
        if (m_pattern[i] < other.m_pattern[i]) return -1;
        if (m_pattern[i] > other.m_pattern[i]) return +1;
    }
    for (auto i = 0u; i < m_last_nodes_incl.size(); ++i) {
        // N.b., needs operator<(cst_XXX::node_type, cst_XXX::node_type) and operator> 
        if (m_last_nodes_incl[i] < other.m_last_nodes_incl[i]) return -1;
        if (m_last_nodes_incl[i] > other.m_last_nodes_incl[i]) return +1;
    }
    return 0;
}

