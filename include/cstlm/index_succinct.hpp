#pragma once

#include "utils.hpp"
#include "collection.hpp"
#include "vocab_uncompressed.hpp"
#include "precomputed_stats.hpp"
#include "constants.hpp"
#include "compressed_counts.hpp"
#include "timings.hpp"
#include "prob_cache.hpp"

#include <sdsl/suffix_arrays.hpp>

#include <future>


namespace cstlm {

using namespace std::chrono;

template <class t_cst, uint32_t t_max_ngram_count = 10, uint32_t t_mgram_cache = 3, uint32_t t_cache_entries = 2000000>
class index_succinct {
public:
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_cst cst_type;

    typedef typename t_cst::csa_type csa_type;
    typedef typename t_cst::node_type node_type;
    typedef typename t_cst::string_type string_type;
    typedef typename csa_type::value_type value_type;
    typedef std::vector<value_type> pattern_type;
    typedef typename pattern_type::const_iterator pattern_iterator;
    typedef compressed_counts<> ccounts_type;
    static constexpr bool byte_alphabet = t_cst::csa_type::alphabet_category::WIDTH == 8;
    typedef vocab_uncompressed<byte_alphabet> vocab_type;
    typedef prob_cache<cst_type,t_mgram_cache,t_cache_entries> prob_cache_type;
public:

private: // data
    cst_type m_cst;
    precomputed_stats m_discounts;
    ccounts_type m_precomputed;
    vocab_type m_vocab;
    prob_cache_type m_static_prob_cache;
public:
    const vocab_type& vocab = m_vocab;
    const cst_type& cst = m_cst;
    const precomputed_stats& discounts = m_discounts;
    const ccounts_type& precomputed = m_precomputed;
    const prob_cache_type& cache = m_static_prob_cache;
public:
    index_succinct() = default;
    index_succinct(index_succinct<t_cst, t_max_ngram_count>&& idx)
    {
        m_vocab = std::move(idx.m_vocab);
        m_cst = std::move(idx.m_cst);
        m_discounts = std::move(idx.m_discounts);
        m_precomputed = std::move(idx.m_precomputed);
        m_static_prob_cache = std::move(idx.m_static_prob_cache);
    }
    index_succinct<t_cst, t_max_ngram_count>& operator=(index_succinct<t_cst, t_max_ngram_count>&& idx)
    {
        m_vocab = std::move(idx.m_vocab);
        m_cst = std::move(idx.m_cst);
        m_discounts = std::move(idx.m_discounts);
        m_precomputed = std::move(idx.m_precomputed);
        m_static_prob_cache = std::move(idx.m_static_prob_cache);
        return (*this);
    }
    index_succinct(collection& col, bool is_mkn = false, bool debug_output = true)
    {
        if (col.file_map.count(KEY_SA) == 0) {
            construct_SA(col);
        }
        auto cst_file = col.path + "/tmp/CST-" + sdsl::util::class_to_hash(m_cst) + ".sdsl";
        if (!utils::file_exists(cst_file)) {
            lm_construct_timer timer("CST");
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            if (col.alphabet == alphabet_type::byte_alphabet)
                cfg.id = "TMPBYTE";
            else
                cfg.id = "TMP";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SA];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXT];
            cfg.file_map[sdsl::conf::KEY_TEXT] = col.file_map[KEY_TEXT];
            construct(m_cst, col.file_map[KEY_TEXT], cfg, 0);
            sdsl::store_to_file(m_cst, cst_file);
        }
        else {
            sdsl::load_from_file(m_cst, cst_file);
        }
        auto discounts_file = col.path + "/tmp/DISCOUNTS-MAXN=" + std::to_string(t_max_ngram_count) + "-BYTE="
            + std::to_string(byte_alphabet) + "-"
            + sdsl::util::class_to_hash(m_discounts) + ".sdsl";
        if (!utils::file_exists(discounts_file)) {
            lm_construct_timer timer("DISCOUNTS");
            m_discounts = precomputed_stats(col, m_cst, t_max_ngram_count);
            sdsl::store_to_file(m_discounts, discounts_file);
            if(debug_output) m_discounts.print(is_mkn, t_max_ngram_count);
        }
        else {
            sdsl::load_from_file(m_discounts, discounts_file);
        }
        auto precomputed_file = col.path + "/tmp/PRECOMPUTED_COUNTS-MAXN="
            + std::to_string(t_max_ngram_count) + "-BYTE="
            + std::to_string(byte_alphabet) + "-"
            + sdsl::util::class_to_hash(m_precomputed) + ".sdsl";
        if (!utils::file_exists(precomputed_file)) {
            lm_construct_timer timer("PRECOMPUTED_COUNTS");
            m_precomputed = ccounts_type(col, m_cst, t_max_ngram_count, is_mkn);
            sdsl::store_to_file(m_precomputed, precomputed_file);
        }
        else {
            sdsl::load_from_file(m_precomputed, precomputed_file);
        }

        {
            m_vocab = vocab_type(col);
        }
        
        // at the end when the index if functional we create the probability
        // cache for low order m-grams
        auto probcache_file = col.path + "/tmp/PROBCACHE-MAXM="
            + std::to_string(t_mgram_cache) + "-MAXC="
            + std::to_string(t_max_ngram_count) + "-BYTE="
            + std::to_string(byte_alphabet) + "-"
            + sdsl::util::class_to_hash(m_static_prob_cache) + ".sdsl";
        if (!utils::file_exists(probcache_file)) {
            lm_construct_timer timer("PROB_CACHE");
            m_static_prob_cache = prob_cache_type(*this);
            std::ofstream ofs(probcache_file);
            m_static_prob_cache.serialize(ofs,m_cst);
        } else {
            std::ifstream ifs(probcache_file);
            m_static_prob_cache.load(ifs,m_cst);
        }
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
        std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_cst.serialize(out, child, "CST");
        written_bytes += m_discounts.serialize(out, child, "Precomputed_Stats");
        written_bytes += m_precomputed.serialize(out, child, "Prestored N1plusfrontback");
        written_bytes += sdsl::serialize(m_vocab, out, child, "Vocabulary");
        written_bytes += m_static_prob_cache.serialize(out,m_cst,child, "Prob_Cache");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    void load(std::istream& in)
    {
        m_cst.load(in);
        sdsl::load(m_discounts, in);
        sdsl::load(m_precomputed, in);
        sdsl::load(m_vocab, in);
        m_static_prob_cache.load(in,m_cst);
    }

    void swap(index_succinct& a)
    {
        if (this != &a) {
            m_cst.swap(a.m_cst);
            std::swap(m_discounts, a.m_discounts);
            std::swap(m_precomputed, a.m_precomputed);
            m_vocab.swap(a.m_vocab);
            m_static_prob_cache.swap(a.m_static_prob_cache);
        }
    }

    uint64_t vocab_size() const
    {
        return m_cst.csa.sigma - 2; // -2 for excluding 0, and 1
    }

    uint64_t N1PlusBack(const node_type& node, pattern_iterator pattern_begin,
        pattern_iterator) const
    {
#ifdef ENABLE_CSTLM_TIMINGS
        auto timer = lm_bench::bench(timer_type::N1PlusBack);
#endif
        uint64_t n1plus_back;
        if (m_cst.is_leaf(node)) {
            n1plus_back = 1;
            // FIXME: does this really follow? Yes, there's only 1 previous context as
            // this node goes to the end of the corpus
        }
        else if (m_cst.depth(node) <= t_max_ngram_count) {
            n1plus_back = m_precomputed.lookup_b(m_cst, node);
        }
        else {
            static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::value_type> preceding_syms(
                m_cst.csa.sigma);
            static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> left(
                m_cst.csa.sigma);
            static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> right(
                m_cst.csa.sigma);

            auto lb = m_cst.lb(node);
            auto rb = m_cst.rb(node);
            typename t_cst::csa_type::size_type num_syms = 0;
            sdsl::interval_symbols(m_cst.csa.wavelet_tree, lb, rb + 1, num_syms,
                preceding_syms, left, right);
            n1plus_back = num_syms;
        }

        // adjust for sentinel start of sentence
        auto symbol = *pattern_begin;
        if (symbol == PAT_START_SYM) {
            n1plus_back -= 1;
        }

        return n1plus_back;
    }

    double discount(uint64_t level, bool cnt = false) const
    {
        // trim to the maximum computed length, assuming that
        // discounts stay flat beyond this (a reasonable guess)
        level = std::min(level, (uint64_t)t_max_ngram_count);
        if (cnt)
            return m_discounts.Y_cnt[level];
        else
            return m_discounts.Y[level];
    }

    void mkn_discount(uint64_t level, double& D1, double& D2, double& D3p,
        bool cnt = false) const
    {
        // trim to the maximum computed length, assuming that
        // discounts stay flat beyond this (a reasonable guess)
        level = std::min(level, (uint64_t)t_max_ngram_count);
        if (cnt) {
            D1 = m_discounts.D1_cnt[level];
            D2 = m_discounts.D2_cnt[level];
            D3p = m_discounts.D3_cnt[level];
        }
        else {
            D1 = m_discounts.D1[level];
            D2 = m_discounts.D2[level];
            D3p = m_discounts.D3[level];
        }
    }

    void print_params(bool ismkn, uint32_t ngramsize) const
    {
        m_discounts.print(ismkn, ngramsize);
    }

    uint64_t N1PlusFrontBack(const node_type& node,
        pattern_iterator pattern_begin,
        pattern_iterator pattern_end) const
    {
#ifdef ENABLE_CSTLM_TIMINGS
        auto timer = lm_bench::bench(timer_type::N1PlusFrontBack);
#endif
        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        if (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node)) {
            if (*pattern_begin == PAT_START_SYM) {
                return m_cst.degree(node);
            }
            else {
                if (pattern_size <= t_max_ngram_count) {
                    return m_precomputed.lookup_fb(m_cst, node);
                }
                else {
                    return compute_contexts(m_cst, node);
                }
            }
        }
        else {
            // special case, only one way of extending this pattern to the right
            if (*pattern_begin == PAT_START_SYM && *(pattern_end - 1) == PAT_END_SYM) {
                /* pattern must be 13xyz41 -> #P(*3xyz4*) == 0 */
                return 0;
            }
            else if (*pattern_begin == PAT_START_SYM) {
                /* pattern must be 13xyzA -> #P(*3xyz*) == 1 */
                return 1;
            }
            else {
                /* pattern must be *xyzA -> #P(*xyz*) == N1PlusBack */
                return N1PlusBack(node, pattern_begin, pattern_end);
            }
        }
    }
    // Computes N_1+( abc * )
    uint64_t N1PlusFront(const node_type& node, pattern_iterator pattern_begin,
        pattern_iterator pattern_end) const
    {
#ifdef ENABLE_CSTLM_TIMINGS
        auto timer = lm_bench::bench(timer_type::N1PlusFront);
#endif
        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        uint64_t N1plus_front;
        if (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node)) {
            // pattern matches the edge label
            N1plus_front = m_cst.degree(node);
        }
        else {
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

    // computes N1(abc *), N_2(abc *), N_3+(abc *) needed for the lower level of
    void N123PlusFrontPrime(const node_type& node, pattern_iterator pattern_begin,
        pattern_iterator pattern_end, uint64_t& f1prime,
        uint64_t& f2prime, uint64_t& f3pprime) const
    {
#ifdef ENABLE_CSTLM_TIMINGS
        auto timer = lm_bench::bench(timer_type::N123PlusFrontPrime);
#endif
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        bool full_match = (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node));
        f1prime = f2prime = f3pprime = 0;
        uint64_t all = 0;
        if (full_match) {
            if (m_precomputed.is_precomputed(m_cst, node)) {
                m_precomputed.lookup_f12prime(m_cst, node, f1prime, f2prime); // FIXME change the name n1plusfrontback
                all = m_cst.degree(node);
            }
            else {
                for (auto child = m_cst.select_child(node, 1); child != m_cst.root(); child = m_cst.sibling(child)) {
                    auto lb = m_cst.lb(child);
                    auto rb = m_cst.rb(child);

                    static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::value_type> preceding_syms(m_cst.csa.sigma);
                    static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> left(m_cst.csa.sigma);
                    static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> right(m_cst.csa.sigma);
                    typename t_cst::csa_type::size_type num_syms = 0;
                    sdsl::interval_symbols(m_cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, left, right);
                    if (num_syms == 1)
                        f1prime++;
                    if (num_syms == 2)
                        f2prime++;
                    all++;
                    child = m_cst.sibling(child);
                }
            }
            f3pprime = all - f1prime - f2prime;
        }
        else {
            // pattern is part of the edge label
            uint64_t num_symsprime = N1PlusBack(node, pattern_begin, pattern_end);
            if (num_symsprime == 1)
                f1prime++;
            if (num_symsprime == 2)
                f2prime++;
            all++; // FIXME: is this right, all is 1? can't see how this might overflow
            f3pprime = all - f1prime - f2prime;
        }
    }

    // Computes N_1( abc * ), N_2( abc * ), N_3+( abc * ); needed for modified
    // Kneser-Ney smoothing
    void N123PlusFront(const node_type& node, pattern_iterator pattern_begin,
        pattern_iterator pattern_end, uint64_t& n1, uint64_t& n2,
        uint64_t& n3p) const
    {
#ifdef ENABLE_CSTLM_TIMINGS
        auto timer = lm_bench::bench(timer_type::N123PlusFront);
#endif
        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        bool full_match = (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node));
        n1 = n2 = n3p = 0;
        if (full_match) {

            if (pattern_size <= t_max_ngram_count) {
                // FIXME: this bit is currently broken
                m_precomputed.lookup_f12(m_cst, node, n1, n2);
                n3p = m_cst.degree(node) - n1 - n2;
            }
            else {
                // loop over the children
                auto child = m_cst.select_child(node, 1);
                auto root = m_cst.root();
                while (child != root) {
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
        }
        else {
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

    uint32_t compute_contexts(const t_cst& cst, const node_type& node) const
    {
        static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::value_type> preceding_syms(
            cst.csa.sigma);
        static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> left(cst.csa.sigma);
        static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> right(
            cst.csa.sigma);
        auto lb = cst.lb(node);
        auto rb = cst.rb(node);
        typename t_cst::csa_type::size_type num_syms = 0;
        sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms,
            preceding_syms, left, right);
        auto total_contexts = 0;
        auto node_depth = cst.depth(node);
        for (size_t i = 0; i < num_syms; i++) {
            auto new_lb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + left[i];
            auto new_rb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + right[i] - 1;
            if (new_lb == new_rb) {
                total_contexts++;
            }
            else {
                auto new_node = cst.node(new_lb, new_rb);

                if (cst.is_leaf(new_node) || cst.depth(new_node) != node_depth + 1) {
                    total_contexts++;
                }
                else {
                    auto deg = cst.degree(new_node);
                    total_contexts += deg;
                }
            }
        }
        return total_contexts;
    }

    uint64_t compute_contexts(const t_cst& cst, const node_type& node,
        uint64_t& count1, uint64_t& count2) const
    {
        static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::value_type> preceding_syms(
            cst.csa.sigma);
        static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> left(cst.csa.sigma);
        static thread_local std::vector<typename t_cst::csa_type::wavelet_tree_type::size_type> right(
            cst.csa.sigma);
        auto lb = cst.lb(node);
        auto rb = cst.rb(node);
        typename t_cst::csa_type::size_type num_syms = 0;
        sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms,
            preceding_syms, left, right);
        auto total_contexts = 0;
        auto node_depth = cst.depth(node);
        count1 = 0;
        count2 = 0;
        for (size_t i = 0; i < num_syms; i++) {
            auto new_lb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + left[i];
            auto new_rb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + right[i] - 1;
            if (new_lb == new_rb) {
                total_contexts++;
                count1++; // size = 1, as [lb, rb] covers single entry
            }
            else {
                auto new_node = cst.node(new_lb, new_rb);
                if (cst.is_leaf(new_node) || cst.depth(new_node) != node_depth + 1) {
                    total_contexts++;
                    // account for count 1 and count 2 entries
                    auto size = cst.size(new_node);
                    if (size == 1)
                        count1++;
                    else if (size == 2)
                        count2++;
                }
                else {
                    auto deg = cst.degree(new_node);
                    total_contexts += deg;
                    // need to know how many of the children have cst.size(new_node) == 1
                    // or 2
                    uint64_t delta1 = 0, delta2 = 0;
                    if (m_precomputed.is_precomputed(cst, new_node)) {
                        // efficient way to compute based on earlier pass computing f1 and
                        // f2 values
                        m_precomputed.lookup_f12(cst, new_node, delta1, delta2);
                    }
                    else {
                        // inefficient way
                        for (const auto& child : cst.children(new_node)) {
                            auto size = cst.size(child);
                            if (size == 1)
                                delta1++;
                            else if (size == 2)
                                delta2++;
                        }
                    }
                    count1 += delta1;
                    count2 += delta2;
                }
            }
        }
        return total_contexts;
    }
};
}
