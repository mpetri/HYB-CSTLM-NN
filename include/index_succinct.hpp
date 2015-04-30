#pragma once

#include "utils.hpp"
#include "collection.hpp"
#include "vocab_uncompressed.hpp"

#include <sdsl/suffix_arrays.hpp>

template <
class t_cst,
class t_vocab = vocab_uncompressed
>
class index_succinct {
public:
    static const int max_ngram_count = 10;
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_cst cst_type;
    typedef t_vocab vocab_type;
    typedef typename t_cst::csa_type csa_type;
    typedef typename t_cst::string_type string_type;
    t_cst m_cst;
    t_cst m_cst_rev;

    std::vector<double> m_n1;
    std::vector<double> m_n2;
    std::vector<double> m_n3;
    std::vector<double> m_n4;
    uint64_t m_N1plus_dotdot;
    uint64_t m_N3plus_dot;
    std::vector<double> m_Y;
    std::vector<double> m_D1;
    std::vector<double> m_D2;
    std::vector<double> m_D3;

    vocab_type m_vocab;

    std::vector<double> m_n1_cnt;
    std::vector<double> m_n2_cnt;
    std::vector<double> m_n3_cnt;
    std::vector<double> m_n4_cnt;
    std::vector<double> m_Y_cnt;
    std::vector<double> m_D1_cnt;
    std::vector<double> m_D2_cnt;
    std::vector<double> m_D3_cnt;


public:
    int
    N1PlusBack(std::vector<uint64_t> pat, bool check_for_EOS = true)
    {
        int pat_size = pat.size();
        uint64_t n1plus_back = 0;
        uint64_t lb_rev = 0, rb_rev = m_cst_rev.size() - 1;
        if (backward_search(m_cst_rev.csa, lb_rev, rb_rev, pat.rbegin(), pat.rend(), lb_rev, rb_rev) > 0) {
            auto node = m_cst_rev.node(lb_rev, rb_rev);
            if (pat_size == m_cst_rev.depth(node)) {
                n1plus_back = m_cst_rev.degree(node);
                if (check_for_EOS) {
                    auto w = m_cst_rev.select_child(node, 1);
                    uint64_t symbol = m_cst_rev.edge(w, pat_size + 1);
                    if (symbol == 1)
                        n1plus_back = n1plus_back - 1;
                }
            } else {
                if (check_for_EOS) {
                    uint64_t symbol = m_cst_rev.edge(node, pat_size + 1);
                    if (symbol != 1)
                        n1plus_back = 1;
                } else {
                    n1plus_back = 1;
                }
            }
        }
        return n1plus_back;
    }

    int
    ActualCount(std::vector<uint64_t>pat)
    {
	uint64_t lb = 0 , rb = m_cst.size()-1;
        if (backward_search(m_cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb) > 0) 
		return rb-lb+1;
	else
		return 0;
    }

    void
    ncomputer(uint64_t symbol, std::vector<uint64_t> pat, int size, uint64_t lb, uint64_t rb)
    {
        auto freq = 0;
        if (lb == rb)
            freq = 1;
        if (size != 0 && lb != rb) {
            freq = rb - lb + 1;
            if (freq == 1 && lb != rb) {
                freq = 0;
            }
        }
        if (size != 0) {
	    pat.push_back(symbol);
            {	
		uint64_t n1plus_back=0;

		if(pat[0]!=3)
                	n1plus_back = N1PlusBack(pat);
		else
			//special case where the pattern starts with <s>: acutal count is used
			n1plus_back = ActualCount(pat);

                if (n1plus_back == 1) {
                    m_n1_cnt[size] += 1;
                } else if (n1plus_back == 2) {
                    m_n2_cnt[size] += 1;
                } else if (n1plus_back == 3) {
                    m_n3_cnt[size] += 1;
                } else if (n1plus_back == 4) {
                    m_n4_cnt[size] += 1;
                }
            }
            if (size == 2 && freq >= 1) {
                m_N1plus_dotdot++;
            }
            if (freq == 1) {
                m_n1[size] += 1;
            } else if (freq == 2) {
                m_n2[size] += 1;
            } else if (freq >= 3) {
                if (freq == 3) {
                    m_n3[size] += 1;
                } else if (freq == 4) { 
                    m_n4[size] += 1;
                }
                if (size == 1)
                    m_N3plus_dot++;
            }
        }
        if (size == 0) {
            auto w = m_cst.select_child(m_cst.root(), 1);
            int root_id = m_cst.id(m_cst.root());
            while (m_cst.id(w) != root_id) {
                symbol = m_cst.edge(w, 1);
                if (symbol != 1 && symbol != 0) {
//                    pat.push_back(symbol);
                    ncomputer(symbol,pat, size + 1, m_cst.lb(w), m_cst.rb(w));
                }
                w = m_cst.sibling(w);
            }
        } else {
            if (size + 1 <= max_ngram_count) {
                if (freq > 0) {
                    auto node = m_cst.node(lb, rb);
                    auto depth = m_cst.depth(node);
                    if (size == depth) {
                        auto w = m_cst.select_child(node, 1);
                        int root_id = m_cst.id(m_cst.root());
                        while (m_cst.id(w) != root_id) {
                            symbol = m_cst.edge(w, depth + 1);
                            if (symbol != 1) {
//                                pat.push_back(symbol);
                                ncomputer(symbol, pat, size + 1, m_cst.lb(w), m_cst.rb(w));
                            }
                            w = m_cst.sibling(w);
                        }
                    } else {
                        symbol = m_cst.edge(node, size + 1);
                        if (symbol != 1) {
//                            pat.push_back(symbol);
                            ncomputer(symbol, pat, size + 1, m_cst.lb(node), m_cst.rb(node));
                        }
                    }
                } else {
                }
            }
        }
    }

    index_succinct() = default;
    index_succinct(collection& col, bool output = true)
    {
        if (output)
            std::cout << "CONSTRUCT CST" << std::endl;
        {
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMP";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SA];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXT];
            construct(m_cst, col.file_map[KEY_TEXT], cfg, 0);
        }
        if (output)
            std::cout << "DONE" << std::endl;
        if (output)
            std::cout << "CONSTRUCT CST REV" << std::endl;
        {
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMPREV";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SAREV];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXTREV];
            construct(m_cst_rev, col.file_map[KEY_TEXTREV], cfg, 0);
        }
        if (output)
            std::cout << "DONE" << std::endl;
        if (output)
            std::cout << "COMPUTE DISCOUNTS" << std::endl;

        m_n1.resize(max_ngram_count + 1);
        m_n2.resize(max_ngram_count + 1);
        m_n3.resize(max_ngram_count + 1);
        m_n4.resize(max_ngram_count + 1);

        m_n1_cnt.resize(max_ngram_count + 1);
        m_n2_cnt.resize(max_ngram_count + 1);
        m_n3_cnt.resize(max_ngram_count + 1);
        m_n4_cnt.resize(max_ngram_count + 1);

        uint64_t lb = 0, rb = m_cst.size() - 1;
	uint64_t symbol=0;
        std::vector<uint64_t> pat;
	m_N1plus_dotdot=0;
	m_N3plus_dot=0;
        ncomputer(symbol, pat, 0, lb, rb);

        m_Y.resize(max_ngram_count + 1);
        m_D1.resize(max_ngram_count + 1);
        m_D2.resize(max_ngram_count + 1);
        m_D3.resize(max_ngram_count + 1);

        m_Y_cnt.resize(max_ngram_count + 1);
        m_D1_cnt.resize(max_ngram_count + 1);
        m_D2_cnt.resize(max_ngram_count + 1);
        m_D3_cnt.resize(max_ngram_count + 1);

        for (int size = 1; size <= max_ngram_count; size++) {
            m_Y[size] = (double)m_n1[size] / (m_n1[size] + 2 * m_n2[size]);
            if (m_n1[size] != 0)
                m_D1[size] = 1 - 2 * m_Y[size] * (double)m_n2[size] / m_n1[size];
            if (m_n2[size] != 0)
                m_D2[size] = 2 - 3 * m_Y[size] * (double)m_n3[size] / m_n2[size];
            if (m_n3[size] != 0)
                m_D3[size] = 3 - 4 * m_Y[size] * (double)m_n4[size] / m_n3[size];
        }

        if(output) std::cout << "DONE" << std::endl;
        if(output) std::cout << "CREATE VOCAB" << std::endl;
       	    m_vocab = vocab_type(col);

        if(output) std::cout << "DONE" << std::endl;

        for (int size = 1; size <= max_ngram_count; size++) {
            m_Y_cnt[size] = (double)m_n1_cnt[size] / (m_n1_cnt[size] + 2 * m_n2_cnt[size]);
            if (m_n1_cnt[size] != 0)
                m_D1_cnt[size] = 1 - 2 * m_Y_cnt[size] * (double)m_n2_cnt[size] / m_n1_cnt[size];
            if (m_n2_cnt[size] != 0)
                m_D2_cnt[size] = 2 - 3 * m_Y_cnt[size] * (double)m_n3_cnt[size] / m_n2_cnt[size];
            if (m_n3_cnt[size] != 0)
                m_D3_cnt[size] = 3 - 4 * m_Y_cnt[size] * (double)m_n4_cnt[size] / m_n3_cnt[size];
        }

        if (output)
            std::cout << "DONE" << std::endl;
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL, std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_cst.serialize(out, child, "CST");
        written_bytes += m_cst_rev.serialize(out, child, "CST_REV");
        written_bytes += sdsl::serialize(m_N1plus_dotdot, out, child, "N1plusdotdot");
        written_bytes += sdsl::serialize(m_N3plus_dot, out, child, "N3plusdot");
        written_bytes += sdsl::serialize(m_n1, out, child, "n1");
        written_bytes += sdsl::serialize(m_n2, out, child, "n2");
        written_bytes += sdsl::serialize(m_n3, out, child, "n3");
        written_bytes += sdsl::serialize(m_n4, out, child, "n4");
        written_bytes += sdsl::serialize(m_Y, out, child, "Y");
        written_bytes += sdsl::serialize(m_D1, out, child, "D1");
        written_bytes += sdsl::serialize(m_D2, out, child, "D2");
        written_bytes += sdsl::serialize(m_D3, out, child, "D3");

        written_bytes += sdsl::serialize(m_vocab, out, child, "Vocabulary");

        written_bytes += sdsl::serialize(m_n1_cnt, out, child, "n1_cnt");
        written_bytes += sdsl::serialize(m_n2_cnt, out, child, "n2_cnt");
        written_bytes += sdsl::serialize(m_n3_cnt, out, child, "n3_cnt");
        written_bytes += sdsl::serialize(m_n4_cnt, out, child, "n4_cnt");
        written_bytes += sdsl::serialize(m_Y_cnt, out, child, "Y_cnt");
        written_bytes += sdsl::serialize(m_D1_cnt, out, child, "D1_cnt");
        written_bytes += sdsl::serialize(m_D2_cnt, out, child, "D2_cnt");
        written_bytes += sdsl::serialize(m_D3_cnt, out, child, "D3_cnt");

        sdsl::structure_tree::add_size(child, written_bytes);

        return written_bytes;
    }

    void load(std::istream& in)
    {
        m_cst.load(in);
        m_cst_rev.load(in);

        sdsl::read_member(m_N1plus_dotdot, in);
        sdsl::read_member(m_N3plus_dot, in);

        sdsl::load(m_n1, in);
        sdsl::load(m_n2, in);
        sdsl::load(m_n3, in);
        sdsl::load(m_n4, in);

        sdsl::load(m_Y, in);
        sdsl::load(m_D1, in);
        sdsl::load(m_D2, in);
        sdsl::load(m_D3, in);

        sdsl::load(m_vocab, in);
        sdsl::load(m_n1_cnt, in);
        sdsl::load(m_n2_cnt, in);
        sdsl::load(m_n3_cnt, in);
        sdsl::load(m_n4_cnt, in);

        sdsl::load(m_Y_cnt, in);
        sdsl::load(m_D1_cnt, in);
        sdsl::load(m_D2_cnt, in);
        sdsl::load(m_D3_cnt, in);
    }

    void swap(index_succinct& a)
    {
        if (this != &a) {
            m_cst.swap(a.m_cst);
            m_cst_rev.swap(a.m_cst_rev);

            std::swap(m_N1plus_dotdot, a.m_N1plus_dotdot);
            std::swap(m_N3plus_dot, a.m_N3plus_dot);

            m_n1_cnt.swap(a.m_n1_cnt);
            m_n2_cnt.swap(a.m_n2_cnt);
            m_n3_cnt.swap(a.m_n3_cnt);
            m_n4_cnt.swap(a.m_n4_cnt);
            m_Y_cnt.swap(a.m_Y_cnt);
            m_D1_cnt.swap(a.m_D1_cnt);
            m_D2_cnt.swap(a.m_D2_cnt);
            m_D3_cnt.swap(a.m_D3_cnt);

            m_n1.swap(a.m_n1);
            m_n2.swap(a.m_n2);
            m_n3.swap(a.m_n3);
            m_n4.swap(a.m_n4);
            m_Y.swap(a.m_Y);
            m_D1.swap(a.m_D1);
            m_D2.swap(a.m_D2);
            m_D3.swap(a.m_D3);
            m_vocab.swap(a.m_vocab);
        }
    }

    uint64_t vocab_size() const
    {
	return m_cst.csa.sigma - 2; // -2 for excluding 0, and 1
    }
};
