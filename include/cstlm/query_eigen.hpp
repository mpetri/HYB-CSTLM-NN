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


public:
    LMQueryMKNE() { m_dest_vocab = nullptr; }
    LMQueryMKNE(const t_idx*                     idx,
                const vocab_uncompressed<false>& vocab,
                uint64_t                         ngramsize,
                bool                             start_sentence = true)
        : m_dest_vocab(&vocab), m_local_state(idx, ngramsize, start_sentence)
    {
    }

    Eigen::VectorXd append_symbol(const uint32_t& symbol)
    {
        Eigen::VectorXd log_prob_vec(m_dest_vocab->size());

        /*
        m_local_state.append_symbol(symbol);
        auto words_following = m_local_state->words_following();
        for (word : words_following) {
            auto mapped_word_id = m_dest_vocab->big2small(word);
            if (word == UNKNOWN_SYM) {
                // TODO UNK HANDLING?
                continue;
            }
            auto state_copy              = m_local_state;
            auto prob                    = state_copy.append_symbol(word);
            log_prob_vec(mapped_word_id) = prob;
        }
        */

        return log_prob_vec;
    }

public:
    const vocab_uncompressed<false>* m_dest_vocab;
    LMQueryMKN<t_idx>                m_local_state;
};
}