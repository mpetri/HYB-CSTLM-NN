#pragma once

#include "utils.hpp"
#include "collection.hpp"
#include "vocab_uncompressed.hpp"
#include "precomputed_stats.hpp"
#include "constants.hpp"
#include "compressed_counts.hpp"
#include "sentinel_flag.hpp"

#include <sdsl/suffix_arrays.hpp>

using namespace std::chrono;

template <class t_cst,
    class t_vocab = vocab_uncompressed,
    uint32_t t_max_ngram_count = 10
    >
class index_succinct_store_n1fb {
public:
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_cst cst_type;
    typedef t_vocab vocab_type;
    typedef typename t_cst::csa_type csa_type;
    typedef typename t_cst::string_type string_type;
public: // data
    t_cst m_cst;
    t_cst m_cst_rev;
    precomputed_stats m_precomputed;
    compressed_counts m_n1plusfrontback;
    vocab_type m_vocab;
    //compressed_sentinel_flag m_csf, m_csf_rev; // trevor: temporary?
public:
    index_succinct_store_n1fb() = default;
    index_succinct_store_n1fb(collection& col)
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
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";

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
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";

        LOG(INFO) << "PRECOMPUTE N1PLUSFRONTBACK";
        start = clock::now();
        m_n1plusfrontback = compressed_counts(m_cst,t_max_ngram_count);
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";

        LOG(INFO) << "COMPUTE DISCOUNTS";
        start = clock::now(); 
        m_precomputed = precompute_statistics(col,m_cst,m_cst_rev,t_max_ngram_count);
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";

        LOG(INFO) << "CREATE VOCAB";
        start = clock::now(); 
        m_vocab = vocab_type(col);
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";

        // perhaps temporary: this and the next block; interested in the relative timing cf 'precompute_statistics'
        //LOG(INFO) << "CREATE EDGE FLAG";
        //start = clock::now(); 
        //m_csf = compressed_sentinel_flag(m_cst);
        //stop = clock::now();
        //LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";

        //LOG(INFO) << "CREATE EDGE FLAG REV";
        //start = clock::now(); 
        //m_csf_rev = compressed_sentinel_flag(m_cst_rev);
        //stop = clock::now();
        //LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL, std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_cst.serialize(out, child, "CST");
        written_bytes += m_cst_rev.serialize(out, child, "CST_REV");
        written_bytes += m_precomputed.serialize(out, child, "Precomputed_Stats");
        written_bytes += m_n1plusfrontback.serialize(out, child, "Prestored N1plusfrontback");
        //written_bytes += m_csf.serialize(out, child, "sentinel");
        //written_bytes += m_csf_rev.serialize(out, child, "sentinel_rev");
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
        sdsl::load(m_vocab, in);
    }

    void swap(index_succinct_store_n1fb& a)
    {
        if (this != &a) {
            m_cst.swap(a.m_cst);
            m_cst_rev.swap(a.m_cst_rev);
            std::swap(m_precomputed,a.m_precomputed);
            std::swap(m_n1plusfrontback,a.m_n1plusfrontback);
            m_vocab.swap(a.m_vocab);
        }
    }

    uint64_t vocab_size() const
    {
        return m_cst.csa.sigma - 2; // -2 for excluding 0, and 1
    }

    uint64_t N1PlusBack(const uint64_t& lb_rev, const uint64_t& rb_rev, uint64_t patrev_size, bool check_for_EOS = true) const
    {
        uint64_t c = 0;
        auto node = m_cst_rev.node(lb_rev, rb_rev);
        if (patrev_size == m_cst_rev.depth(node)) {
            c = m_cst_rev.degree(node);
            if (check_for_EOS) {
                auto w = m_cst_rev.select_child(node, 1);
                uint64_t symbol = m_cst_rev.edge(w, patrev_size + 1);
                if (symbol == EOS_SYM)
                    c = c - 1;
            }
        } else {
            if (check_for_EOS) {
                uint64_t symbol = m_cst_rev.edge(node, patrev_size + 1);
                if (symbol != EOS_SYM)
                    c = 1;
            } else {
                c = 1;
            }
        }
        return c;
    }

    double discount(uint64_t level, bool cnt = false) const
    {
        if (cnt)
            return m_precomputed.Y_cnt[level];
        else
            return m_precomputed.Y[level];
    }

    void print_params(bool ismkn,uint32_t ngramsize) const {
        m_precomputed.print(ismkn,ngramsize);
    }

    //  Computes N_1+( * ab * )
    //  n1plus_front = value of N1+( * abc ) (for some following symbol 'c')
    //  if this is N_1+( * ab ) = 1 then we know the only following symbol is 'c'
    //  and thus N1+( * ab * ) is the same as N1+( * abc ), stored in n1plus_back
    uint64_t N1PlusFrontBack(const uint64_t& lb, const uint64_t& rb,
                             const uint64_t n1plus_back,
                             const std::vector<uint64_t>::iterator& pattern_begin,
                             const std::vector<uint64_t>::iterator& pattern_end,
                             bool check_for_EOS = true) const
    {

        auto node = m_cst.node(lb, rb);
        return m_n1plusfrontback.lookup(m_cst,node);
    }

    // Computes N_1+( abc * )
    uint64_t N1PlusFront(const uint64_t& lb, const uint64_t& rb,
                         std::vector<uint64_t>::iterator pattern_begin,
                         std::vector<uint64_t>::iterator pattern_end,
                         bool check_for_EOS = true) const
    {
        // ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
        auto node = m_cst.node(lb, rb);
        auto pattern_size = std::distance(pattern_begin, pattern_end);
        uint64_t N1plus_front = 0;
        if (pattern_size == m_cst.depth(node)) {
            auto w = m_cst.select_child(node, 1);
            N1plus_front = m_cst.degree(node);
            if (check_for_EOS) {
                uint64_t symbol = m_cst.edge(w, pattern_size + 1);
                if (symbol == EOS_SYM) {
                    N1plus_front = N1plus_front - 1;
                }
            }
            return N1plus_front;
        } else {
            if (check_for_EOS) {
                uint64_t symbol = m_cst.edge(node, pattern_size + 1);
                if (symbol != EOS_SYM) {
                    N1plus_front = 1;
                }
            }
            return N1plus_front;
        }
    }

};
