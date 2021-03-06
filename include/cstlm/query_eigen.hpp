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

    const vector_type& append_symbol(const uint32_t& symbol)
    {
        m_local_state.append_symbol(symbol);

        auto           cur_hash        = m_local_state.hash();
        const uint64_t max_cache_elems = 500000;
        static std::map<uint64_t, uint64_t> cache_stats;
        {
            auto itr = local_cache.find(cur_hash);
            if (itr != local_cache.end()) {
                cache_stats[cur_hash]++;
                return itr->second;
            }
        }

        /* compute if we can't find it */
        static vector_type log_prob_vec(m_dest_vocab->size(), -99);
        //auto wordsfollowing = m_local_state.words_following();
        for (const auto& word_itr : *m_dest_vocab) {
            auto word           = word_itr.second;
            auto mapped_word_id = m_dest_vocab->small2big(word);
            if (word != UNKNOWN_SYM && mapped_word_id == UNKNOWN_SYM) {
                log_prob_vec[mapped_word_id] = -99;
                std::cerr << "TODO UNK IN BIG BUT NOT IN SMALL???? " << word << " - "
                          << mapped_word_id << std::endl;
                continue;
            }
            auto state_copy    = m_local_state;
            auto logprob       = state_copy.append_symbol(mapped_word_id);
            log_prob_vec[word] = logprob;
        }

        // add to cache if it is a bit more complex to compute
        if (local_cache.size() < max_cache_elems) {
            cache_stats[cur_hash] = 1;
            local_cache[cur_hash] = log_prob_vec;
            return local_cache[cur_hash];
        } else {
            auto smallest         = cache_stats.begin();
            auto smallest_id      = smallest->first;
            local_cache[cur_hash] = log_prob_vec;
            cache_stats[cur_hash] = 1;
            local_cache.erase(smallest_id);
            cache_stats.erase(smallest);
            return local_cache[cur_hash];
        }
        return log_prob_vec;
    }

public:
    const vocab_uncompressed<false>* m_dest_vocab;
    LMQueryMKN<t_idx>                m_local_state;
    std::unordered_map<uint64_t, vector_type>& local_cache;
};
}
