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

namespace cstlm {

// Returns the Kneser-Ney probability of a sentence, word at a
// time. Words are supplied using the append_symbol method which
// returns the conditional probability of that word given all
// previous words.

template <class t_idx>
class LMQueryMKN {
public:
    using value_type = typename t_idx::value_type;
    using node_type = typename t_idx::cst_type::node_type;
    using index_type = t_idx;

public:
    LMQueryMKN()
    {
        m_idx = nullptr;
    }
    LMQueryMKN(const index_type* idx, uint64_t ngramsize, bool start_sentence = true);
    double append_symbol(const value_type& symbol);

    bool operator==(const LMQueryMKN& other) const;

    size_t hash() const;

    bool empty() const
    {
        return m_last_nodes_incl.size() == 1 && m_last_nodes_incl.back() == m_idx->cst.root();
    }

    bool is_start() const
    {
        return m_pattern.size() == 1 && m_pattern.back() == PAT_START_SYM;
    }

private:
    const index_type* m_idx;
    uint64_t m_ngramsize;
    std::vector<node_type> m_last_nodes_incl;
    std::deque<value_type> m_pattern;
};

template <class t_idx>
LMQueryMKN<t_idx>::LMQueryMKN(const t_idx* idx, uint64_t ngramsize, bool start_sentence)
    : m_idx(idx)
    , m_ngramsize(ngramsize)
{
    auto root = m_idx->cst.root();
    m_last_nodes_incl.push_back(root);
    if (start_sentence) {
        auto node = root;
        auto r = backward_search_wrapper(*m_idx, node, PAT_START_SYM);
        (void)r;
        assert(r >= 0);
        m_last_nodes_incl.push_back(node);
        m_pattern.push_back(PAT_START_SYM);
    }
}

template <class t_idx>
double LMQueryMKN<t_idx>::append_symbol(const value_type& symbol)
{
    if (symbol == PAT_START_SYM && m_pattern.size() == 1 && m_pattern.front() == PAT_START_SYM)
        return log10(1);

    m_pattern.push_back(symbol);
    while (m_ngramsize > 0 && m_pattern.size() > m_ngramsize)
        m_pattern.pop_front();
    std::vector<value_type> pattern(m_pattern.begin(), m_pattern.end());

    // fast way, tracking state
    double p = 1.0 / (m_idx->vocab.size() - 4);
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

        double D1, D2, D3p;
        m_idx->mkn_discount(i, D1, D2, D3p, i == 1 || i != m_ngramsize);

        double c, d;
        if ((i == m_ngramsize && m_ngramsize != 1) || (*start == PAT_START_SYM)) {
            c = (ok) ? m_idx->cst.size(node_incl) : 0;
            d = m_idx->cst.size(node_excl);
        }
        else if (i == 1 || m_ngramsize == 1) {
            c = (ok) ? m_idx->N1PlusBack(node_incl, start, pattern_end) : 0;
            d = m_idx->discounts.N1plus_dotdot;
        }
        else {
            c = (ok) ? m_idx->N1PlusBack(node_incl, start, pattern_end) : 0;
            d = m_idx->N1PlusFrontBack(node_excl, start, pattern_end - 1);
        }

        if (c == 1) {
            c -= D1;
        }
        else if (c == 2) {
            c -= D2;
        }
        else if (c >= 3) {
            c -= D3p;
        }

        uint64_t n1 = 0, n2 = 0, n3p = 0;
        if ((i == m_ngramsize && m_ngramsize != 1) || (*start == PAT_START_SYM)) {
            m_idx->N123PlusFront(node_excl, start, pattern_end - 1, n1, n2, n3p);
        }
        else if (i == 1 || m_ngramsize == 1) {
            n1 = m_idx->discounts.n1_cnt[1];
            n2 = m_idx->discounts.n2_cnt[1];
            n3p = (m_idx->vocab_size() - 2) - (n1 + n2);
        }
        else {
            m_idx->N123PlusFrontPrime(node_excl, start, pattern_end - 1, n1, n2, n3p);
        }

        // n3p is dodgy
        double gamma = D1 * n1 + D2 * n2 + D3p * n3p;
        p = (c + gamma * p) / d;

        //LOG(INFO) << "\t\ti = " << i << " p = " << p << " c = " << c << " gamma " << gamma << " d = " << d;
        //LOG(INFO) << "\t\t\t" << D1 << ":" << n1 << ":" << D2 << ":" << n2 << ":" << D3p << ":" << n3p;
    }

    m_last_nodes_incl = node_incl_vec;
    while (m_pattern.size() > m_last_nodes_incl.size())
        m_pattern.pop_front();

    return log10(p);
}

template <class t_idx>
bool LMQueryMKN<t_idx>::operator==(const LMQueryMKN& other) const
{
    if (m_idx != other.m_idx)
        return false;
    if (m_pattern.size() != other.m_pattern.size())
        return false;
    if (m_last_nodes_incl.size() != other.m_last_nodes_incl.size())
        return false;
    for (auto i = 0u; i < m_pattern.size(); ++i) {
        if (m_pattern[i] != other.m_pattern[i])
            return false;
    }
    for (auto i = 0u; i < m_last_nodes_incl.size(); ++i) {
        if (m_last_nodes_incl[i] != other.m_last_nodes_incl[i])
            return false;
    }
    return true;
}

template <class t_idx>
std::size_t LMQueryMKN<t_idx>::hash() const
{
    std::size_t seed = 0;
    for (auto& i : m_pattern) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    for (auto i = 0u; i < m_last_nodes_incl.size(); ++i) {
        auto id = m_idx->cst.id(m_last_nodes_incl[i]);
        seed ^= id + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
}