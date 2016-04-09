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
#ifdef STATELESS_QUERY
#include "kn.hpp"
#endif
namespace cstlm {

//#define STATELESS_QUERY

// Returns the Kneser-Ney probability of a sentence, word at a
// time. Words are supplied using the append_symbol method which
// returns the conditional probability of that word given all
// previous words.

template <class t_idx>
class LMQueryKN {
    using value_type = typename t_idx::value_type;
    using node_type = typename t_idx::cst_type::node_type;
    using index_type = t_idx;

public:
    LMQueryKN()
    {
        m_idx = nullptr;
    }
    LMQueryKN(const t_idx* idx, uint64_t ngramsize);
    double append_symbol(const value_type& symbol);
    int compare(const LMQueryKN& other) const;

private:
    const index_type* m_idx;
    uint64_t m_ngramsize;
    std::vector<node_type> m_last_nodes_incl;
    std::deque<value_type> m_pattern;
};

template <class t_idx>
LMQueryKN<t_idx>::LMQueryKN(const t_idx* idx, uint64_t ngramsize)
    : m_idx(idx)
    , m_ngramsize(ngramsize)
{
    auto root = m_idx->cst.root();
    auto node = root;
    auto r = backward_search_wrapper(*m_idx, node, PAT_START_SYM);
    (void)r;
    assert(r >= 0);
    m_last_nodes_incl = std::vector<node_type>({ root, node });
    m_pattern.push_back(PAT_START_SYM);
}

template <class t_idx>
double LMQueryKN<t_idx>::append_symbol(const value_type& symbol)
{
    if (symbol == PAT_START_SYM && m_pattern.size() == 1 && m_pattern.front() == PAT_START_SYM)
        return 1;

    m_pattern.push_back(symbol);
    while (m_ngramsize > 0 && m_pattern.size() > m_ngramsize)
        m_pattern.pop_front();
    std::vector<value_type> pattern(m_pattern.begin(), m_pattern.end());
#ifdef STATELESS_QUERY
    // slow way
    return prob_kneser_ney(*m_idx, pattern.begin(), pattern.end(), m_ngramsize);
#else
    // fast way, tracking state
    double p = 1.0;
    node_type node_incl = m_idx->cst.root(); // v_F^all matching the full pattern, including last item
    auto node_excl_it = m_last_nodes_incl.begin(); // v_F     matching only the context, excluding last item
    node_type node_excl = *node_excl_it;
    auto pattern_begin = pattern.begin();
    auto pattern_end = pattern.end();

    size_t size = std::distance(pattern_begin, pattern_end);
    bool unk = (*(pattern_end - 1) == UNKNOWN_SYM);
    bool ok = !unk;
    std::vector<node_type> node_incl_vec({ node_incl });

    for (unsigned i = 1; i <= size; ++i) {
        auto start = pattern_end - i;
        if (i > 1 && *start == UNKNOWN_SYM)
            break;
        if (ok) {
            ok = backward_search_wrapper(*m_idx, node_incl, *start);
            if (ok)
                node_incl_vec.push_back(node_incl);
        }

        // recycle the node_incl matches from the last call to append_symbol
        // to serve as the node_excl values
        if (i >= 2) {
            node_excl_it++;
            if (node_excl_it == m_last_nodes_incl.end()) {
                break;
            }
            else {
                node_excl = *node_excl_it;
            }
        }

        double D = m_idx->discount(i, i == 1 || i != m_ngramsize);
        double c, d;
        if ((i == m_ngramsize && m_ngramsize != 1) || (*start == PAT_START_SYM)) {
            c = (ok) ? m_idx->cst.size(node_incl) : 0;
            d = m_idx->cst.size(node_excl);
        }
        else if (i == 1 || m_ngramsize == 1) {
            c = (ok) ? m_idx->N1PlusBack(node_incl, start, pattern_end) : D;
            d = m_idx->discounts.N1plus_dotdot;
        }
        else {
            c = (ok) ? m_idx->N1PlusBack(node_incl, start, pattern_end) : 0;
            d = m_idx->N1PlusFrontBack(node_excl, start, pattern_end - 1);
        }

        // update the running probability
        if (i > 1) {
            double q = m_idx->N1PlusFront(node_excl, start, pattern_end - 1);
            p = (std::max(c - D, 0.0) + D * q * p) / d;
        }
        else {
            p = c / d;
        }
    }

    m_last_nodes_incl = node_incl_vec;
    while (m_pattern.size() > m_last_nodes_incl.size())
        m_pattern.pop_front();

    return p;
#endif
}

template <class t_idx>
int LMQueryKN<t_idx>::compare(const LMQueryKN& other) const
{
    if (m_idx < other.m_idx)
        return -1;
    if (m_idx > other.m_idx)
        return +1;
    if (m_pattern.size() < other.m_pattern.size())
        return -1;
    if (m_pattern.size() > other.m_pattern.size())
        return +1;
    if (m_last_nodes_incl.size() < other.m_last_nodes_incl.size())
        return -1;
    if (m_last_nodes_incl.size() > other.m_last_nodes_incl.size())
        return +1;
    for (auto i = 0u; i < m_pattern.size(); ++i) {
        if (m_pattern[i] < other.m_pattern[i])
            return -1;
        if (m_pattern[i] > other.m_pattern[i])
            return +1;
    }
    for (auto i = 0u; i < m_last_nodes_incl.size(); ++i) {
        // N.b., needs operator<(cst_XXX::node_type, cst_XXX::node_type) and
        // operator>
        if (m_last_nodes_incl[i] < other.m_last_nodes_incl[i])
            return -1;
        if (m_last_nodes_incl[i] > other.m_last_nodes_incl[i])
            return +1;
    }
    return 0;
}
}