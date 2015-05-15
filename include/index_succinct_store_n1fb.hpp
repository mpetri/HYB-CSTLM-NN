#pragma once

#include "utils.hpp"
#include "collection.hpp"
#include "vocab_uncompressed.hpp"
#include "precomputed_stats.hpp"
#include "constants.hpp"
#include "compressed_counts.hpp"
//#include "sentinel_flag.hpp"

#include <sdsl/suffix_arrays.hpp>

using namespace std::chrono;

template <class t_cst, class t_vocab = vocab_uncompressed, uint32_t t_max_ngram_count = 10>
class index_succinct_store_n1fb {
public:
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_cst cst_type;
    typedef t_vocab vocab_type;
    typedef typename t_cst::csa_type csa_type;
    typedef typename t_cst::node_type node_type;
    typedef typename t_cst::string_type string_type;
    typedef std::vector<uint64_t> pattern_type;
    typedef typename pattern_type::const_iterator pattern_iterator;

public: // data
    t_cst m_cst;
    t_cst m_cst_rev;
    precomputed_stats m_precomputed;
    compressed_counts<> m_n1plusfrontback;
    vocab_type m_vocab;
    // compressed_sentinel_flag<> m_csf, m_csf_rev; // trevor: temporary?
public:
    index_succinct_store_n1fb() = default;
    index_succinct_store_n1fb(collection& col, bool dodgy_discounts=false)
    {
        using clock = std::chrono::high_resolution_clock;

        auto start = clock::now();
        LOG(INFO) << "CONSTRUCT CST";
        {
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMP";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SA];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXT];
            construct(m_cst, col.file_map[KEY_TEXT], cfg, 0);
        }
        auto stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f
                  << " sec)";

        LOG(INFO) << "CONSTRUCT CST REV";
        start = clock::now();
        {
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMPREV";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SAREV];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXTREV];
            construct(m_cst_rev, col.file_map[KEY_TEXTREV], cfg, 0);
        }
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f
                  << " sec)";

        LOG(INFO) << "PRECOMPUTE N1PLUSFRONTBACK";
        start = clock::now();
        m_n1plusfrontback = compressed_counts<>(m_cst, t_max_ngram_count);
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f
                  << " sec)";

        LOG(INFO) << "COMPUTE DISCOUNTS";
        start = clock::now();
        m_precomputed = precomputed_stats(col, m_cst_rev, t_max_ngram_count, dodgy_discounts);
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f
                  << " sec)";

        // m_precomputed.print(false, 10);

        LOG(INFO) << "CREATE VOCAB";
        start = clock::now();
        m_vocab = vocab_type(col);
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f
                  << " sec)";

        // perhaps temporary: this and the next block; interested in the relative timing cf
        // 'precompute_statistics'
        //        LOG(INFO) << "CREATE EDGE FLAG";
        //        start = clock::now();
        //        m_csf = compressed_sentinel_flag<>(m_cst);
        //        stop = clock::now();
        //        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() /
        //        1000.0f << " sec)";
        //
        //        LOG(INFO) << "CREATE EDGE FLAG REV";
        //        start = clock::now();
        //        m_csf_rev = compressed_sentinel_flag<>(m_cst_rev);
        //        stop = clock::now();
        //        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() /
        //        1000.0f << " sec)";
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
                        std::string name = "") const
    {
        sdsl::structure_tree_node* child
            = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_cst.serialize(out, child, "CST");
        written_bytes += m_cst_rev.serialize(out, child, "CST_REV");
        written_bytes += m_precomputed.serialize(out, child, "Precomputed_Stats");
        written_bytes += m_n1plusfrontback.serialize(out, child, "Prestored N1plusfrontback");
        // written_bytes += m_csf.serialize(out, child, "sentinel");
        // written_bytes += m_csf_rev.serialize(out, child, "sentinel_rev");
        written_bytes += sdsl::serialize(m_vocab, out, child, "Vocabulary");

        sdsl::structure_tree::add_size(child, written_bytes);

        return written_bytes;
    }

    void load(std::istream& in)
    {
        m_cst.load(in);
        m_cst_rev.load(in);
        sdsl::load(m_precomputed, in);
        sdsl::load(m_n1plusfrontback, in);
        // sdsl::load(m_csf, in);
        // sdsl::load(m_csf_rev, in);
        sdsl::load(m_vocab, in);
    }

    void swap(index_succinct_store_n1fb& a)
    {
        if (this != &a) {
            m_cst.swap(a.m_cst);
            m_cst_rev.swap(a.m_cst_rev);
            std::swap(m_precomputed, a.m_precomputed);
            std::swap(m_n1plusfrontback, a.m_n1plusfrontback);
            // std::swap(m_csf, a.m_csf);
            // std::swap(m_csf_rev, a.m_csf_rev);
            m_vocab.swap(a.m_vocab);
        }
    }

    uint64_t vocab_size() const
    {
        return m_cst.csa.sigma - 2; // -2 for excluding 0, and 1
    }

    uint64_t N1PlusBack(const node_type &node_rev, 
            pattern_iterator pattern_begin, pattern_iterator pattern_end) const
    {
        // ASSUMPTION: node_rev matches the pattern in the reverse tree, m_cst_rev
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);

        uint64_t n1plus_back;
        if (!m_cst_rev.is_leaf(node_rev) && pattern_size == m_cst_rev.depth(node_rev)) {
            n1plus_back = m_cst_rev.degree(node_rev);
        } else {
            n1plus_back = 1;
        }

        // adjust for sentinel start of sentence
        auto symbol = *pattern_begin;
        if (symbol == PAT_START_SYM)
            n1plus_back -= 1;

        return n1plus_back;
    }

    double discount(uint64_t level, bool cnt = false) const
    {
        if (cnt)
            return m_precomputed.Y_cnt[level];
        else
            return m_precomputed.Y[level];
    }

    void print_params(bool ismkn, uint32_t ngramsize) const
    {
        m_precomputed.print(ismkn, ngramsize);
    }

    //  Computes N_1+( * ab * )
    //  n1plus_front = value of N1+( * abc ) (for some following symbol 'c')
    //  if this is N_1+( * ab ) = 1 then we know the only following symbol is 'c'
    //  and thus N1+( * ab * ) is the same as N1+( * abc ), stored in n1plus_back
    uint64_t N1PlusFrontBack(const node_type &node, const node_type &node_rev,
                             pattern_iterator pattern_begin, pattern_iterator pattern_end) const
    {
        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        // ASSUMPTION: node_rev matches the pattern in the reverse tree, m_cst_rev
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        if (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node)) {
            if (*pattern_begin == PAT_START_SYM) {
                return m_cst.degree(node);
            } else {
                return m_n1plusfrontback.lookup(m_cst, node);
            }
        } else {
            // special case, only one way of extending this pattern to the right
            if (*pattern_begin == PAT_START_SYM && *(pattern_end - 1) == PAT_END_SYM) {
                /* pattern must be 13xyz41 -> #P(*3xyz4*) == 0 */
                return 0;
            } else if (*pattern_begin == PAT_START_SYM) {
                /* pattern must be 13xyzA -> #P(*3xyz*) == 1 */
                return 1;
            } else {
                /* pattern must be *xyzA -> #P(*xyz*) == N1PlusBack */
                return N1PlusBack(node_rev, pattern_begin, pattern_end);
            }
        }
    }

    // Computes N_1+( abc * )
    uint64_t N1PlusFront(const node_type &node, 
            pattern_iterator pattern_begin, pattern_iterator pattern_end) const
    {
        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        uint64_t N1plus_front;
        if (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node)) {
            // pattern matches the edge label
            N1plus_front = m_cst.degree(node);
        } else {
            // pattern is part of the edge label
            N1plus_front = 1;
        }

        // adjust for end of sentence
        uint64_t symbol = *(pattern_end - 1);
        if (symbol == PAT_END_SYM) {
            N1plus_front -= 1;
        }
        return N1plus_front;
    }
};
