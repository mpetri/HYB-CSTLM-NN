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
#include <unordered_map>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"
#include "constants.hpp"

#include "logging.hpp"

#include <Eigen/Dense>

namespace cstlm {

template <class t_idx>
class LMQueryMKNE {
    using eigen_type = Eigen::Matrix<float, Eigen::Dynamic, 1>;

public:
    LMQueryMKNE() { m_dest_vocab = nullptr; }
    LMQueryMKNE(const t_idx*                     idx,
                const vocab_uncompressed<false>& vocab,
                uint64_t                         ngramsize,
                bool                             start_sentence = true)
        : m_dest_vocab(&vocab), m_local_state(idx, ngramsize, start_sentence)
    {
    }

    eigen_type append_symbol(const uint32_t& symbol)
    {
        m_local_state.append_symbol(symbol);

        /* cache first */
        static std::unordered_map<uint64_t, eigen_type> local_cache;
        auto cur_hash = m_local_state.hash();
        auto itr      = local_cache.find(cur_hash);
        if (itr != local_cache.end()) {
            std::cout << "PROB_CACHE_HIT('" << m_local_state.cur_node_label() << "')" << std::endl;
            return itr->second;
        }

        /* compute if we can't find it */
        Eigen::Matrix<float, Eigen::Dynamic, 1> log_prob_vec(m_dest_vocab->size());
        log_prob_vec.fill(0);
        auto wordsfollowing = m_local_state.words_following();
        for (const auto& word : wordsfollowing) {
            auto mapped_word_id = m_dest_vocab->big2small(word);
            if (word == UNKNOWN_SYM) {
                // TODO UNK HANDLING?
                continue;
            }
            auto state_copy              = m_local_state;
            auto prob                    = state_copy.append_symbol(word);
            log_prob_vec(mapped_word_id) = prob;
        }

        // add to cache if it is a bit more complex to compute
        if (wordsfollowing.size() > 10) {
            local_cache[cur_hash] = log_prob_vec;
        }
        return log_prob_vec;
    }

public:
    const vocab_uncompressed<false>* m_dest_vocab;
    LMQueryMKN<t_idx>                m_local_state;
};
}