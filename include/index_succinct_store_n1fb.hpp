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

template <class t_cst, class t_cst_rev, class t_vocab = vocab_uncompressed, uint32_t t_max_ngram_count = 10>
class index_succinct_store_n1fb {
public:
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_cst cst_type;
    typedef t_cst_rev cst_rev_type;
    typedef t_vocab vocab_type;
    typedef typename t_cst::csa_type csa_type;
    typedef typename t_cst::node_type node_type;
    typedef typename t_cst::string_type string_type;
    typedef std::vector<uint64_t> pattern_type;
    typedef typename pattern_type::const_iterator pattern_iterator;
    static const bool supports_forward_querying = true;

public: // data
    t_cst m_cst;
    t_cst_rev m_cst_rev;
    precomputed_stats m_precomputed;
    compressed_counts<> m_n1plusfrontback;
    vocab_type m_vocab;
public:
    index_succinct_store_n1fb() = default;
    index_succinct_store_n1fb(collection& col,bool is_mkn=false)
    {
        {
            lm_construct_timer timer("CST_REV");
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMPREV";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SAREV];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXTREV];
            construct(m_cst_rev, col.file_map[KEY_TEXTREV], cfg, 0);
        }
        {
            lm_construct_timer timer("DISCOUNTS");
            m_precomputed = precomputed_stats(col, m_cst_rev, t_max_ngram_count, is_mkn);
        }
        {
            lm_construct_timer timer("CST");
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMP";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SA];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXT];
            construct(m_cst, col.file_map[KEY_TEXT], cfg, 0);
        }
        {
            lm_construct_timer timer("PRECOMPUTED_COUNTS");
            m_n1plusfrontback = compressed_counts<>(m_cst, t_max_ngram_count, is_mkn);
        }
        {
            lm_construct_timer timer("VOCAB");
            m_vocab = vocab_type(col);
        }
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
                        std::string name = "") const
    {
        sdsl::structure_tree_node* child
            = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_cst.serialize(out, child, "CST");
        written_bytes += m_precomputed.serialize(out, child, "Precomputed_Stats");
        written_bytes += m_n1plusfrontback.serialize(out, child, "Prestored N1plusfrontback");
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
            m_cst_rev.swap(a.m_cst);
            std::swap(m_precomputed, a.m_precomputed);
            std::swap(m_n1plusfrontback, a.m_n1plusfrontback);
            m_vocab.swap(a.m_vocab);
        }
    }

    uint64_t vocab_size() const
    {
        return m_cst.csa.sigma - 2; // -2 for excluding 0, and 1
    }

    uint64_t N1PlusBack_from_forward(const node_type &node,
            pattern_iterator pattern_begin, pattern_iterator pattern_end) const
    {
        auto timer = lm_bench::bench(timer_type::N1PlusBack);

        //std::cout << "N1PlusBack_from_forward -- pattern ";
        //std::copy(pattern_begin, pattern_end, std::ostream_iterator<uint64_t>(std::cout, " "));
        //std::cout << std::endl;
        //std::cout << "\tnode is " << node << " root is " << m_cst.root() << std::endl;

        uint64_t n1plus_back;
        if (m_cst.is_leaf(node)) {
            //std::cout << "\tleaf\n";
            n1plus_back = 1;
            // FIXME: does this really follow? Yes, there's only 1 previous context as this node goes to the end of the corpus
        } else if (m_cst.depth(node) <= t_max_ngram_count) {
            n1plus_back = m_n1plusfrontback.lookup_b(m_cst, node);
            //std::cout << "\tnon-leaf\n";
        } else {
            //std::cout << "\tdepth exceeded\n";
            // when depth is exceeded, we don't precompute the N1+FB/N1+B scores
            // so we need to compute these explictly

            static std::vector<typename t_cst::csa_type::value_type> preceding_syms(m_cst.csa.sigma);
            static std::vector<typename t_cst::csa_type::size_type> left(m_cst.csa.sigma);
            static std::vector<typename t_cst::csa_type::size_type> right(m_cst.csa.sigma);

            auto lb = m_cst.lb(node);
            auto rb = m_cst.rb(node);
            typename t_cst::csa_type::size_type num_syms = 0;
            sdsl::interval_symbols(m_cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, 
                    left, right);
            n1plus_back = num_syms;
        }

        // adjust for sentinel start of sentence
        auto symbol = *pattern_begin;
        if (symbol == PAT_START_SYM) {
            //std::cout << "\tpat start decrement\n";
            n1plus_back -= 1;
        }
        //std::cout << "N1PlusBack_from_forward returning: " << n1plus_back << "\n";

        return n1plus_back;
    }

    uint64_t N1PlusBack(const node_type &node_rev,
            pattern_iterator pattern_begin, pattern_iterator pattern_end) const
    {
        auto timer = lm_bench::bench(timer_type::N1PlusBack);

        // FIXME: if pattern is longer than t_max_ngram_count we may not have prestored counts so we should compute them explicitly.
        
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
        // trim to the maximum computed length, assuming that
        // discounts stay flat beyond this (a reasonable guess)
        level = std::min(level, (uint64_t) t_max_ngram_count);
        if (cnt)
            return m_precomputed.Y_cnt[level];
        else
            return m_precomputed.Y[level];
    }

    void mkn_discount(uint64_t level, double &D1, double &D2, double &D3p, bool cnt = false) const
    {
        // trim to the maximum computed length, assuming that
        // discounts stay flat beyond this (a reasonable guess)
        level = std::min(level, (uint64_t) t_max_ngram_count);
        if (cnt) {
            D1 = m_precomputed.D1_cnt[level];
            D2 = m_precomputed.D2_cnt[level];
            D3p = m_precomputed.D3_cnt[level];
        } else {
            D1 = m_precomputed.D1[level];
            D2 = m_precomputed.D2[level];
            D3p = m_precomputed.D3[level];
        }
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
        auto timer = lm_bench::bench(timer_type::N1PlusFrontBack);

        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        // ASSUMPTION: node_rev matches the pattern in the reverse tree, m_cst_rev
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        if (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node)) {
            if (*pattern_begin == PAT_START_SYM) {
                return m_cst.degree(node);
            } else {
                return m_n1plusfrontback.lookup_fb(m_cst, node);
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

    uint64_t N1PlusFrontBack_from_forward(const node_type &node,
                             pattern_iterator pattern_begin, pattern_iterator pattern_end) const
    {
        auto timer = lm_bench::bench(timer_type::N1PlusFrontBack);

        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        // ASSUMPTION: node_rev matches the pattern in the reverse tree, m_cst_rev
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        if (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node)) {
            if (*pattern_begin == PAT_START_SYM) {
                return m_cst.degree(node);
            } else {
                return m_n1plusfrontback.lookup_fb(m_cst, node);
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
                return N1PlusBack_from_forward(node, pattern_begin, pattern_end);
            }
        }
    }

    // Computes N_1+( abc * )
    uint64_t N1PlusFront(const node_type &node, 
            pattern_iterator pattern_begin, pattern_iterator pattern_end) const
    {
        auto timer = lm_bench::bench(timer_type::N1PlusFront);

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
    
    // Computes N_1( abc * ), N_2( abc * ), N_3+( abc * ); needed for modified Kneser-Ney smoothing
    void N123PlusFront(const node_type &node,
                       pattern_iterator pattern_begin, pattern_iterator pattern_end,
                       uint64_t &n1, uint64_t &n2, uint64_t &n3p) const
    {
        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        bool full_match = (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node));
        n1 = n2 = n3p = 0;
        if (full_match) {
            if (pattern_size <= t_max_ngram_count) {
                // FIXME: this bit is currently broken
                m_n1plusfrontback.lookup_f12(m_cst, node, n1, n2);
                n3p = m_cst.degree(node) - n1 - n2;
            } else {
                // loop over the children
                auto child = m_cst.select_child(node, 1); 
                while (child != m_cst.root()) {
                    auto c = m_cst.size(child);
                    if (c == 1)
                        n1 += 1;
                    else if (c == 2)
                        n2 += 1;
                    else if (c >= 3)
                        n3p += 1;
                    child = m_cst.sibling(child);
                }
            }
        } else {
            // pattern is part of the edge label
            uint64_t symbol = *(pattern_end - 1);
            if (symbol != PAT_END_SYM) {
                auto c = m_cst.size(node);
                if (c == 1)
                    n1 += 1;
                else if (c == 2)
                    n2 += 1;
                else if (c >= 3)
                    n3p += 1;
            } 
        }
    }
};
