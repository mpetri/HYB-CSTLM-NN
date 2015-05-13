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
    typedef std::vector<uint64_t> pattern_type;
    typedef typename pattern_type::const_iterator pattern_iterator;

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
        m_precomputed = precomputed_stats(col, m_cst, m_cst_rev, t_max_ngram_count);
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

    uint64_t N1PlusBack(uint64_t lb_rev, uint64_t rb_rev, 
                         pattern_iterator pattern_begin,
                         pattern_iterator pattern_end) const
    {
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        auto node = m_cst_rev.node(lb_rev, rb_rev);

        uint64_t n1plus_back;
        if (pattern_size == m_cst_rev.depth(node)) {
            n1plus_back = m_cst_rev.degree(node);
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
    uint64_t N1PlusFrontBack(uint64_t lb, uint64_t rb,
                         uint64_t lb_rev, uint64_t rb_rev,
                         pattern_iterator pattern_begin,
                         pattern_iterator pattern_end) const
    {
        // ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
        // ASSUMPTION: pattern_begin, pattern_end cover just the pattern we're interested in (i.e., we want N1+ dot pattern dot)
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        auto node = m_cst.node(lb, rb);
        uint64_t back_N1plus_front = 0;
        uint64_t lb_rev_stored = 0, rb_rev_stored = 0;
        // this is when the pattern matches a full edge in the CST
        if (pattern_size == m_cst.depth(node)) {
            if (*pattern_begin == PAT_START_SYM ) {
                return m_cst.degree(node);
            }
            auto w = m_cst.select_child(node, 1);
            auto root_id = m_cst.id(m_cst.root());
            std::vector<uint64_t> new_pattern(pattern_begin, pattern_end);
            new_pattern.push_back(EOS_SYM);
            while (m_cst.id(w) != root_id) {
                lb_rev_stored = lb_rev;
                rb_rev_stored = rb_rev;
                uint64_t symbol = m_cst.edge(w, pattern_size + 1);
                assert(symbol != EOS_SYM);
                new_pattern.back() = symbol;
                // find the symbol to the right
                // (which is first in the reverse order)
                backward_search(m_cst_rev.csa,
                                lb_rev_stored, rb_rev_stored,
                                symbol,
                                lb_rev_stored, rb_rev_stored);

                back_N1plus_front += N1PlusBack(lb_rev_stored, rb_rev_stored, 
                        new_pattern.begin(), new_pattern.end());
                w = m_cst.sibling(w);
            }
            return back_N1plus_front;
        } else {
            // special case, only one way of extending this pattern to the right
            if (*pattern_begin == PAT_START_SYM
                    && *(pattern_end-1) == PAT_END_SYM) {
                /* pattern must be 13xyz41 -> #P(*3xyz4*) == 0 */
                return 0;
            } else if (*pattern_begin == PAT_START_SYM) {
                /* pattern must be 13xyzA -> #P(*3xyz*) == 1 */
                return 1;
            } else {
                /* pattern must be *xyzA -> #P(*xyz*) == N1PlusBack */
                return N1PlusBack(lb_rev, rb_rev, pattern_begin, pattern_end);
            }
        }
    }

    // Computes N_1+( abc * )
    uint64_t N1PlusFront(uint64_t lb, uint64_t rb,
                         pattern_iterator pattern_begin,
                         pattern_iterator pattern_end) const
    {
        // ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
        auto node = m_cst.node(lb, rb);
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        uint64_t N1plus_front;
        if (pattern_size == m_cst.depth(node)) {
            // pattern matches the edge label
            N1plus_front = m_cst.degree(node);
        } else {
            // pattern is part of the edge label
            N1plus_front = 1;
        }

        // adjust for end of sentence 
        uint64_t symbol = *(pattern_end-1);
        if (symbol == PAT_END_SYM) {
            N1plus_front -= 1;
        }
        return N1plus_front;
    }
};
