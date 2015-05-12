#pragma once

#include "utils.hpp"
#include "collection.hpp"
#include "constants.hpp"
#include "vocab_uncompressed.hpp"
#include "precomputed_stats.hpp"
#include "constants.hpp"

#include <sdsl/suffix_arrays.hpp>

using namespace std::chrono;

template <class t_cst,
          class t_vocab = vocab_uncompressed,
          uint32_t t_max_ngram_count = 10>
class index_succinct {
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
    vocab_type m_vocab;
public:
    index_succinct() = default;
    index_succinct(collection& col)
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

        LOG(INFO) << "COMPUTE DISCOUNTS";
        start = clock::now();
        m_precomputed = precompute_statistics(col, m_cst, m_cst_rev, t_max_ngram_count);
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";

        LOG(INFO) << "CREATE VOCAB";
        start = clock::now();
        m_vocab = vocab_type(col);
        stop = clock::now();
        LOG(INFO) << "DONE (" << duration_cast<milliseconds>(stop - start).count() / 1000.0f << " sec)";
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL, std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_cst.serialize(out, child, "CST");
        written_bytes += m_cst_rev.serialize(out, child, "CST_REV");
        written_bytes += m_precomputed.serialize(out, child, "Precomputed_Stats");
        written_bytes += sdsl::serialize(m_vocab, out, child, "Vocabulary");

        sdsl::structure_tree::add_size(child, written_bytes);

        return written_bytes;
    }

    void load(std::istream& in)
    {
        m_cst.load(in);
        m_cst_rev.load(in);
        sdsl::load(m_precomputed, in);
        sdsl::load(m_vocab, in);
    }

    void swap(index_succinct& a)
    {
        if (this != &a) {
            m_cst.swap(a.m_cst);
            m_cst_rev.swap(a.m_cst_rev);
            std::swap(m_precomputed, a.m_precomputed);
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

    void print_params(bool ismkn, uint32_t ngramsize) const
    {
        m_precomputed.print(ismkn, ngramsize);
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
        // ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
        // ASSUMPTION: pattern_begin, pattern_end cover just the pattern we're interested in (i.e., we want N1+ dot pattern dot)
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        auto node = m_cst.node(lb, rb);
        uint64_t back_N1plus_front = 0;
        uint64_t lb_rev_prime = 0, rb_rev_prime = m_cst_rev.size() - 1;
        uint64_t lb_rev_stored = 0, rb_rev_stored = 0;
        // this is a full search for the pattern in reverse order in the reverse tree!
        for (auto it = pattern_begin; it != pattern_end and lb_rev_prime <= rb_rev_prime;) {
            backward_search(m_cst_rev.csa,
                            lb_rev_prime, rb_rev_prime,
                            *it,
                            lb_rev_prime, rb_rev_prime);
            it++;
        }
        // this is when the pattern matches a full edge in the CST
        if (pattern_size == m_cst.depth(node)) {
            auto w = m_cst.select_child(node, 1);
            auto root_id = m_cst.id(m_cst.root());
            while (m_cst.id(w) != root_id) {
                lb_rev_stored = lb_rev_prime;
                rb_rev_stored = rb_rev_prime;
                uint64_t symbol = m_cst.edge(w, pattern_size + 1);
                if (symbol != EOS_SYM || !check_for_EOS) {
                    // find the symbol to the right
                    // (which is first in the reverse order)
                    backward_search(m_cst_rev.csa,
                                    lb_rev_stored, rb_rev_stored,
                                    symbol,
                                    lb_rev_stored, rb_rev_stored);

                    back_N1plus_front += N1PlusBack(lb_rev_stored, rb_rev_stored, pattern_size + 1, check_for_EOS);
                }
                w = m_cst.sibling(w);
            }
            return back_N1plus_front;
        } else {
            // special case, only one way of extending this pattern to the right
            return n1plus_back;
        }
    }

    // Computes N_1+( abc * )
    uint64_t N1PlusFront(const uint64_t& lb, const uint64_t& rb,
                         std::vector<uint64_t>::iterator pattern_begin,
                         std::vector<uint64_t>::iterator pattern_end,
                         bool check_for_EOS = true) const
    {
        // ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
        auto node = m_cst.node(lb, rb);
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        uint64_t N1plus_front = 0;
        if (pattern_size == m_cst.depth(node)) {
            N1plus_front = m_cst.degree(node);
            if (check_for_EOS) {
                auto w = m_cst.select_child(node, 1);
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
