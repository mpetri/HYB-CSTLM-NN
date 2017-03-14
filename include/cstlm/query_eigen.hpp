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
    using vector_type = std::vector<float>;

public:
    LMQueryMKNE() { m_dest_vocab = nullptr; }
    LMQueryMKNE(const t_idx*                     idx,
                const vocab_uncompressed<false>& vocab,
                uint64_t                         ngramsize,
                bool /*start_sentence*/,
                std::unordered_map<uint64_t, std::vector<float>>& cache)
        : m_dest_vocab(&vocab), m_local_state(idx, ngramsize, false), local_cache(cache)
    {
    }

    LMQueryMKNE(const LMQueryMKNE& other)
        : m_dest_vocab(other.m_dest_vocab)
        , m_local_state(other.m_local_state)
        , local_cache(other.local_cache)
    {
    }

    vector_type append_symbol(const uint32_t& symbol)
    {
        m_local_state.append_symbol(symbol);

        auto cur_hash = m_local_state.hash();
        {
            auto itr = local_cache.find(cur_hash);
            if (itr != local_cache.end()) {
                return itr->second;
            }
        }

        /* compute if we can't find it */
        vector_type log_prob_vec(m_dest_vocab->size(), -99);
        //auto wordsfollowing = m_local_state.words_following();
        for (const auto& word_itr : *m_dest_vocab) {
            auto word           = word_itr.second;
            auto mapped_word_id = m_dest_vocab->small2big(word);
            if (word != UNKNOWN_SYM && mapped_word_id == UNKNOWN_SYM) {
                std::cerr << "TODO UNK IN BIG BUT NOT IN SMALL???? " << word << " - "
                          << mapped_word_id << std::endl;
                continue;
            }
            auto state_copy    = m_local_state;
            auto logprob       = state_copy.append_symbol(mapped_word_id);
            log_prob_vec[word] = logprob;
        }

        // add to cache if it is a bit more complex to compute
        {
            local_cache[cur_hash] = log_prob_vec;
        }
        return log_prob_vec;
    }

public:
    const vocab_uncompressed<false>* m_dest_vocab;
    LMQueryMKN<t_idx>                m_local_state;
    std::unordered_map<uint64_t, vector_type>& local_cache;
};
}
