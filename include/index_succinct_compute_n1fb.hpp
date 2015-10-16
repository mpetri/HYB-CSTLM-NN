#pragma once

#include "utils.hpp"
#include "collection.hpp"
#include "vocab_uncompressed.hpp"
#include "precomputed_stats.hpp"
#include "constants.hpp"

#include <sdsl/suffix_arrays.hpp>

using namespace std::chrono;

template <class t_cst, class t_cst_rev, class t_vocab = vocab_uncompressed, uint32_t t_max_ngram_count = 10>
class index_succinct_compute_n1fb {
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
    cst_type m_cst;
    cst_rev_type m_cst_rev;
    precomputed_stats m_precomputed;
    vocab_type m_vocab;

public:
    index_succinct_compute_n1fb() = default;
    index_succinct_compute_n1fb(collection& col, bool is_mkn = false)
    {
        auto cst_rev_file = col.path + "/tmp/CST_REV-" + sdsl::util::class_to_hash(m_cst_rev) + ".sdsl";
        if (!utils::file_exists(cst_rev_file)) {
            lm_construct_timer timer("CST_REV");
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMPREV";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SAREV];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXTREV];
            construct(m_cst_rev, col.file_map[KEY_TEXTREV], cfg, 0);
            sdsl::store_to_file(m_cst_rev, cst_rev_file);
        } else {
            sdsl::load_from_file(m_cst_rev, cst_rev_file);
        }
        auto discounts_file = col.path + "/tmp/DISCOUNTS-" + sdsl::util::class_to_hash(m_precomputed) + ".sdsl";
        if (!utils::file_exists(discounts_file)) {
            lm_construct_timer timer("DISCOUNTS");
            m_precomputed = precomputed_stats(col, m_cst_rev, t_max_ngram_count, is_mkn);
            sdsl::store_to_file(m_precomputed, discounts_file);
        } else {
            sdsl::load_from_file(m_precomputed, discounts_file);
        }
        auto cst_file = col.path + "/tmp/CST-" + sdsl::util::class_to_hash(m_cst) + ".sdsl";
        if (!utils::file_exists(cst_file)) {
            lm_construct_timer timer("CST");
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMP";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SA];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXT];
            construct(m_cst, col.file_map[KEY_TEXT], cfg, 0);
            sdsl::store_to_file(m_cst, cst_file);
        } else {
            sdsl::load_from_file(m_cst, cst_file);
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

    void swap(index_succinct_compute_n1fb& a)
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

    uint64_t N1PlusBack(const node_type& node_rev,
                        pattern_iterator pattern_begin, pattern_iterator pattern_end) const
    {
        auto timer = lm_bench::bench(timer_type::N1PlusBack);

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

    uint64_t N1PlusBack_from_forward(const node_type& node,
                                     pattern_iterator pattern_begin, pattern_iterator) const
    {
        static std::vector<typename t_cst::csa_type::value_type> preceding_syms(m_cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::size_type> left(m_cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::size_type> right(m_cst.csa.sigma);

        //        std::cout << "N1PlusBack_from_forward -- pattern ";
        //        std::copy(pattern_begin, pattern_end, std::ostream_iterator<uint64_t>(std::cout, " "));
        //        std::cout << std::endl;

        auto timer = lm_bench::bench(timer_type::N1PlusBack);
        auto lb = m_cst.lb(node);
        auto rb = m_cst.rb(node);
        typename t_cst::csa_type::size_type num_syms = 0;
        // FIXME: seems wasteful to query for preceding_syms, left, right and then ignore em
        // might want to retain the (left, right) anyway for the next call to backwardsearch
        sdsl::interval_symbols(m_cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms,
                               left, right);

        // adjust for sentinel start of sentence
        auto symbol = *pattern_begin;
        if (symbol == PAT_START_SYM)
            return num_syms - 1;
        else
            return num_syms;
    }

    double discount(uint64_t level, bool cnt = false) const
    {
        // trim to the maximum computed length, assuming that
        // discounts stay flat beyond this (a reasonable guess)
        level = std::min(level, (uint64_t)t_max_ngram_count);
        if (cnt)
            return m_precomputed.Y_cnt[level];
        else
            return m_precomputed.Y[level];
    }

    void mkn_discount(uint64_t level, double& D1, double& D2, double& D3p, bool cnt = false) const
    {
        // trim to the maximum computed length, assuming that
        // discounts stay flat beyond this (a reasonable guess)
        level = std::min(level, (uint64_t)t_max_ngram_count);
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

    uint32_t compute_contexts(const t_cst& cst, const node_type& node) const
    {
        static std::vector<typename t_cst::csa_type::value_type> preceding_syms(cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::size_type> left(cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::size_type> right(cst.csa.sigma);
        auto lb = cst.lb(node);
        auto rb = cst.rb(node);
        typename t_cst::csa_type::size_type num_syms = 0;
        sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, left,
                               right);
        auto total_contexts = 0;
        auto node_depth = cst.depth(node);
        for (size_t i = 0; i < num_syms; i++) {
            auto new_lb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + left[i];
            auto new_rb = cst.csa.C[cst.csa.char2comp[preceding_syms[i]]] + right[i] - 1;
            if (new_lb == new_rb) {
                total_contexts++;
            } else {
                auto new_node = cst.node(new_lb, new_rb);

                if (cst.is_leaf(new_node) || cst.depth(new_node) != node_depth + 1) {
                    total_contexts++;
                } else {
                    auto deg = cst.degree(new_node);
                    total_contexts += deg;
                }
            }
        }
        return total_contexts;
    }

    //  Computes N_1+( * ab * )
    //  n1plus_front = value of N1+( * abc ) (for some following symbol 'c')
    //  if this is N_1+( * ab ) = 1 then we know the only following symbol is 'c'
    //  and thus N1+( * ab * ) is the same as N1+( * abc ), stored in n1plus_back
    uint64_t N1PlusFrontBack(const node_type& node, const node_type& node_rev,
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
                return compute_contexts(m_cst, node);
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

    uint64_t N1PlusFrontBack_from_forward(const node_type& node,
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
                return compute_contexts(m_cst, node);
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
    uint64_t N1PlusFront(const node_type& node,
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

    // computes N1(abc *), N_2(abc *), N_3+(abc *) needed for the lower level of MKN
    void N123PlusFront_lower(const node_type& node,
                             pattern_iterator pattern_begin, pattern_iterator pattern_end,
                             uint64_t& n1, uint64_t& n2, uint64_t& n3p) const
    {
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        bool full_match = (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node));
        n1 = n2 = n3p = 0;
        uint64_t all = 0;
        if (full_match) {
            // pattern matches the edge label
            auto child = m_cst.select_child(node, 1);
            while (child != m_cst.root()) {
                uint64_t symbol = m_cst.edge(child, pattern_size + 1);
                auto lb = m_cst.lb(child);
                auto rb = m_cst.rb(child);

                static std::vector<typename t_cst::csa_type::value_type> preceding_syms(m_cst.csa.sigma);
                static std::vector<typename t_cst::csa_type::size_type> left(m_cst.csa.sigma);
                static std::vector<typename t_cst::csa_type::size_type> right(m_cst.csa.sigma);
                typename t_cst::csa_type::size_type num_syms = 0;
                sdsl::interval_symbols(m_cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, left, right);
                if (num_syms == 1)
                    n1++;
                if (num_syms == 2)
                    n2++;
                all++;
                child = m_cst.sibling(child);
            }
        } else {
            // pattern is part of the edge label
            auto lb = m_cst.lb(node);
            auto rb = m_cst.rb(node);

            static std::vector<typename t_cst::csa_type::value_type> preceding_syms(m_cst.csa.sigma);
            static std::vector<typename t_cst::csa_type::size_type> left(m_cst.csa.sigma);
            static std::vector<typename t_cst::csa_type::size_type> right(m_cst.csa.sigma);
            typename t_cst::csa_type::size_type num_syms = 0;
            sdsl::interval_symbols(m_cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, left, right);
            if (num_syms == 1)
                n1++;
            if (num_syms == 2)
                n2++;
            all++;
        }
        n3p = all - n1 - n2;
    }

    void N123PlusFront(const node_type& node,
                       pattern_iterator pattern_begin, pattern_iterator pattern_end,
                       uint64_t& n1, uint64_t& n2, uint64_t& n3p) const
    {
        //LOG(INFO) << "N123PlusFront -- node " << node << " pattern " << std::vector<uint64_t>(pattern_begin, pattern_end);

        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        bool full_match = (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node));
        n1 = n2 = n3p = 0;
        if (full_match) {
            // pattern matches the edge label
            auto child = m_cst.select_child(node, 1);
            while (child != m_cst.root()) {
                auto c = m_cst.size(child);
                //LOG(INFO) << "\ttop -- child " << child << " count " << c;
                if (c == 1)
                    n1 += 1;
                else if (c == 2)
                    n2 += 1;
                else if (c >= 3)
                    n3p += 1;
                child = m_cst.sibling(child);
            }
            //n1p = m_cst.degree(node);
        } else {
            // pattern is part of the edge label
            uint64_t symbol = *(pattern_end - 1);
            if (symbol != PAT_END_SYM) {
                auto c = m_cst.size(node);
                //LOG(INFO) << "\tbottom -- count " << c;
                if (c == 1)
                    n1 += 1;
                else if (c == 2)
                    n2 += 1;
                else if (c >= 3)
                    n3p += 1;
            } else {
                //LOG(INFO) << "\tbottom -- skipping";
            }
        }
        //LOG(INFO) << "\treturning " << n1 << " " << n2 << " " << n3p;
    }

    void N123PlusBack_from_forward(const node_type &node,
		      pattern_iterator pattern_begin, pattern_iterator pattern_end,
		      uint64_t &n1, uint64_t &n2, uint64_t &n3p) const
    {
	auto timer = lm_bench::bench(timer_type::N1PlusBack);

	n1 = n2 = n3p = 0;
	auto size = m_cst.size(node);
	if (m_cst.is_leaf(node)) {
	    // there's only 1 previous context as this node goes to the end of the corpus
	    if (size == 1)
		n1 = 1;
	    else if (size == 2)
		n2 = 1;
	    else
		n3p = 1;
	} else {
	    static std::vector<typename t_cst::csa_type::value_type> preceding_syms(m_cst.csa.sigma);
	    static std::vector<typename t_cst::csa_type::size_type> left(m_cst.csa.sigma);
	    static std::vector<typename t_cst::csa_type::size_type> right(m_cst.csa.sigma);
	    
	    auto lb = m_cst.lb(node);
	    auto rb = m_cst.rb(node);
	    typename t_cst::csa_type::size_type num_syms = 0;
	    sdsl::interval_symbols(m_cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms,
				   left, right);
	    
	    for (size_t i = 0; i < num_syms; i++) {
		auto new_lb = m_cst.csa.C[m_cst.csa.char2comp[preceding_syms[i]]] + left[i];
		auto new_rb = m_cst.csa.C[m_cst.csa.char2comp[preceding_syms[i]]] + right[i] - 1;
		auto new_size = (new_rb - new_lb + 1);
		if (new_size == 1)
		    n1 += 1;
		else if (new_size == 2)
		    n2 += 1;
		else
		    n3p += 1;
	    }
	}
	
	// adjust for sentinel start of sentence
	auto symbol = *pattern_begin;
	if (symbol == PAT_START_SYM) {
	    if (size == 1)
		n1 -= 1;
	    else if (size == 2)
		n2 -= 1;
	    else
		n3p -= 1;
	}
    }

    void N123PlusFrontBack_from_forward(const node_type &node,
					    pattern_iterator pattern_begin, pattern_iterator pattern_end,
					    uint64_t &n1, uint64_t &n2, uint64_t &n3p) const
    {
        auto timer = lm_bench::bench(timer_type::N123PlusFrontBack);

        // ASSUMPTION: node matches the pattern in the forward tree, m_cst
        // ASSUMPTION: node_rev matches the pattern in the reverse tree, m_cst_rev
        uint64_t pattern_size = std::distance(pattern_begin, pattern_end);
        if (!m_cst.is_leaf(node) && pattern_size == m_cst.depth(node)) {
            if (*pattern_begin == PAT_START_SYM) {
                N123PlusFront(node, pattern_begin, pattern_end, n1, n2, n3p);
            } else {
                auto n1p = compute_contexts(m_cst, node, n1, n2);
                n3p = n1p - n1 - n2;
            }
        } else {
            // special case, only one way of extending this pattern to the right
            if (*pattern_begin == PAT_START_SYM && *(pattern_end - 1) == PAT_END_SYM) {
                /* pattern must be 13xyz41 -> #P(*3xyz4*) == 0 */
                n1 = n2 = n3p = 0;
            } else if (*pattern_begin == PAT_START_SYM) {
                /* pattern must be 13xyzA -> #P(*3xyz*) == 1 */
		auto size = m_cst.size(node);
		n1 = n2 = n3p = 0;
		if (size == 1)
		    n1 = 1;
		else if (size == 2)
		    n2 = 1;
		else
		    n3p = 1;
            } else {
                /* pattern must be *xyzA -> #P(*xyz*) == N1PlusBack */
                N123PlusBack_from_forward(node, pattern_begin, pattern_end, n1, n2, n3p);
            }
        }
    }


    uint64_t compute_contexts(const t_cst& cst, const node_type &node, 
			      uint64_t &count1, uint64_t &count2) const
    {
	static std::vector<typename t_cst::csa_type::value_type> preceding_syms(cst.csa.sigma);
	static std::vector<typename t_cst::csa_type::size_type> left(cst.csa.sigma);
	static std::vector<typename t_cst::csa_type::size_type> right(cst.csa.sigma);
	auto lb = cst.lb(node);
	auto rb = cst.rb(node);
        typename t_cst::csa_type::size_type num_syms = 0;
	sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, left, right);
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
	    } else {
		auto new_node = cst.node(new_lb, new_rb);
		if (cst.is_leaf(new_node) || cst.depth(new_node) != node_depth + 1) {
		    total_contexts++;
		    // account for count 1 and count 2 entries
		    auto size = cst.size(new_node);
		    if (size == 1)
			count1++;
		    else if (size == 2)
			count2++;
		} else {
		    auto deg = cst.degree(new_node);
		    total_contexts += deg;
		    // need to know how many of the children have cst.size(new_node) == 1 or 2
		    uint64_t delta1 = 0, delta2 = 0;
                    for (const auto& child : cst.children(new_node)) {
                        auto size = cst.size(child);
                        if (size == 1)
                            delta1++;
                        else if (size == 2)
                            delta2++;
                    }
                    //LOG(INFO) << " INEFF; node " << new_node << " delta1 " << delta1 << " delta2 " << delta2;
		    count1 += delta1;
		    count2 += delta2;
		}
	    }
	}
	return total_contexts;
    }
};
