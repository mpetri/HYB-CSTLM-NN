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
uint64_t sample_next_symbol(const t_idx& idx, t_pat_iter pattern_begin,
                            t_pat_iter pattern_end, uint64_t ngramsize,
                            t_rng& rng)
{
    return _sample_next_symbol(idx, pattern_begin, pattern_end, ngramsize,
                               idx.m_cst.root(), 0, rng);
}

template <class t_idx, class t_pat_iter, class t_rng>
uint64_t _sample_next_symbol(const t_idx& idx, t_pat_iter pattern_begin,
                             t_pat_iter pattern_end, uint64_t ngramsize,
                             typename t_idx::cst_type::node_type node,
                             uint64_t ctxsize, t_rng& rng)
{
    size_t size = std::distance(pattern_begin, pattern_end);

    LOG(INFO) << "sampling for pattern "
              << std::vector<uint64_t>(pattern_begin, pattern_end) << " ctxsize "
              << ctxsize;

    // attempt a longer match
    uint64_t next = EOF_SYM;
    if (ctxsize < size) {
        t_pat_iter start = pattern_end - (ctxsize + 1);
        if (*start != UNKNOWN_SYM) {
            auto ok = backward_search_wrapper(idx, node, *start);
            if (ok) {
                next = _sample_next_symbol(idx, pattern_begin, pattern_end, ngramsize,
                                           node, ctxsize + 1, rng);
                if (next != EOF_SYM)
                    return next;
            }
        }
    }

    // either longer match failed, or we're at maximum length
    auto i = ctxsize + 1;
    double D = idx.discount(i, i == 1 || i != ngramsize);
    double d;
    auto start = (pattern_end - i + 1);
    if ((i == ngramsize && ngramsize != 1) || (*start == PAT_START_SYM)) {
        d = idx.m_cst.size(node);
    } else if (i == 1 || ngramsize == 1) {
        d = idx.m_discounts.N1plus_dotdot;
    } else {
        d = idx.N1PlusFrontBack(node, start, pattern_end);
    }

    LOG(INFO) << "\tctxsize " << ctxsize << " start " << *start << " d " << d;

    if (i > 1) {
        double q = idx.N1PlusFront(node, start, pattern_end);
        double stay = 1.0 - (D * q) / d;
        LOG(INFO) << "\tq " << q << " stay " << stay;

        std::uniform_real_distribution<> uniform(0, 1);
        double r = uniform(rng);
        LOG(INFO) << "\tflip " << r;
        if (r < stay) {
            double r = uniform(rng) * (d - D * q);
            LOG(INFO) << "\tchild " << r << " of " << d;
            // read off the symbol from the corresponding edge
            auto child = idx.m_cst.select_child(node, 1);
            while (child != idx.m_cst.root()) {
                if ((i == ngramsize) || (*start == PAT_START_SYM)) // condition seems fishy
                    r -= idx.m_cst.size(child) - D;
                else {
                    // augmented pattern is a bit fishy, may overrun memory
                    r -= idx.N1PlusBack(child, start, pattern_end + 1) - D;
                }

                LOG(INFO) << "\t\tr now " << r << " after child " << child
                          << " of size " << idx.m_cst.size();
                if (r <= 0) {
                    // is i the right index or are we off by one? think this is ok, as
                    // it's 1-indexed
                    next = idx.m_cst.edge(child, i);
                    break;
                }
                child = idx.m_cst.sibling(child);
            }
            assert(false && "you shouldn't reach this line");
        } else {
            // backoff
            LOG(INFO) << "\tbacking off";
            next = EOF_SYM;
        }
    } else {
        // FIXME: somewhat wasteful, could be done in one iterator
        static std::vector<uint64_t> unigrams_cs(unigram_counts(idx));
        static std::discrete_distribution<uint64_t> unigrams(unigrams_cs.begin(),
                                                             unigrams_cs.end());
        unigrams_cs.clear();
        next = unigrams(rng);
        LOG(INFO) << "\tsampled unigram";
    }
    LOG(INFO) << "next is " << next;

    return next;
}

template <class t_idx>
std::vector<uint64_t> unigram_counts(const t_idx& idx)
{
    // FIXME: this should be precomputed; and perhaps we can do this faster?
    auto root = idx.m_cst.root();
    std::vector<uint64_t> pattern(1, NUM_SPECIAL_SYMS); // place-holder pattern
    std::vector<uint64_t> weights;
    uint64_t i = 0;
    for (const auto& child : idx.m_cst.children(root)) {
        if (i >= NUM_SPECIAL_SYMS || i == UNKNOWN_SYM || i == PAT_END_SYM) {
            pattern[0] = i;
            weights.push_back(idx.N1PlusBack(child, pattern.begin(), pattern.end()));
        } else {
            weights.push_back(0);
        }
        ++i;
    }
    return weights;
}

template <class t_idx, class t_pat_iter, class t_rng>
uint64_t sample_next_symbol2(const t_idx& idx, t_pat_iter pattern_begin,
                             t_pat_iter pattern_end, uint64_t ngramsize,
                             t_rng& rng)
{
    // size_t size = std::distance(pattern_begin, pattern_end);

    LOG(INFO) << "sampling for pattern "
              << std::vector<uint64_t>(pattern_begin, pattern_end);

    std::vector<double> probs;
    std::vector<uint64_t> pattern(pattern_begin, pattern_end);
    pattern.push_back(EOF_SYM);
    double total = 0;
    for (uint64_t next = 0; next < idx.m_vocab.size(); ++next) {
        if (next >= NUM_SPECIAL_SYMS || next == PAT_END_SYM || next == UNKNOWN_SYM) {
            pattern.back() = next;
            auto prob = prob_kneser_ney_single(idx, pattern.begin(), pattern.end(),
                                               ngramsize);
            total += prob;
        } else {
            probs.push_back(0);
        }
    }

    std::discrete_distribution<uint64_t> pr_next(probs.begin(), probs.end());
    auto next = pr_next(rng);
    LOG(INFO) << "next is " << next;

    return next;
}
