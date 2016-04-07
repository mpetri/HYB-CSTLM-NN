#pragma once

#include "utils.hpp"
#include "collection.hpp"
#include "vocab_uncompressed.hpp"
#include "precomputed_stats.hpp"
#include "constants.hpp"
#include "compressed_counts.hpp"
#include "timings.hpp"
//#include "sentinel_flag.hpp"

#include <sdsl/suffix_arrays.hpp>

using namespace std::chrono;

template <class t_csa_rev, class t_vocab = vocab_uncompressed>
class index_stupid {
public:
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_csa_rev csa_rev_type;
    typedef t_vocab vocab_type;

public: // data
    csa_rev_type m_csa_rev;
    vocab_type m_vocab;

public:
    index_stupid(collection& col)
    {
        auto csa_file = col.path + "/index/CSA_REV-" + sdsl::util::class_to_hash(m_csa_rev) + ".sdsl";
        if (!utils::file_exists(csa_file)) {
            lm_construct_timer timer("CSA_REV");
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMP";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SAREV];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXTREV];
            construct(m_csa_rev, col.file_map[KEY_TEXTREV], cfg, 0);
            sdsl::store_to_file(m_csa_rev, csa_file);
        }
        else {
            sdsl::load_from_file(m_csa_rev, csa_file);
        }

        {
            lm_construct_timer timer("VOCAB");
            m_vocab = vocab_type(col);
        }
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
        std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_csa_rev.serialize(out, child, "CSA_REV");
        written_bytes += sdsl::serialize(m_vocab, out, child, "Vocabulary");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    void load(std::istream& in)
    {
        m_csa_rev.load(in);
        m_vocab.load(in);
    }

    void swap(index_stupid& a)
    {
        if (this != &a) {
            m_csa_rev.swap(a.m_csa_rev);
            m_vocab.swap(a.m_vocab);
        }
    }
};
