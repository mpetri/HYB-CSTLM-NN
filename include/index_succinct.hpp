#pragma once

#include "utils.hpp"
#include "collection.hpp"

#include <sdsl/suffix_arrays.hpp>

template <class t_cst>
class index_succinct {
public:
    const int max_ngram_count = 20;
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_cst cst_type;
    typedef typename t_cst::csa_type csa_type;
    typedef typename t_cst::string_type;
    t_cst m_cst;
    t_cst m_cst_rev;

    std::vector<int> m_n1; 
    std::vector<int> m_n2; 
    std::vector<int> m_n3; 
    std::vector<int> m_n4; 
    int m_N1plus_dotdot;
    int m_N3plus_dot;
    std::vector<double> m_Y;
    std::vector<double> m_D1;
    std::vector<double> m_D2;
    std::vector<double> m_D3;

public:
    int ncomputer(cst_sct3<csa_sada_int<> >::string_type pat, int size, uint64_t lb, uint64_t rb)
    {
        backward_search(m_cst.csa, lb, rb, pat.begin(), pat.end(), lb, rb);
        freq = rb - lb + 1;
        if (freq == 1 && lb != rb) {
            freq = 0;
        }
        if (size != 0) {
            if (pat.size() == 2 && freq >= 1) {
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
            int ind = 0;
            pat.resize(1);
            int degree=m_cst.degree(m_cst.root());
            while (ind < degree) {
                auto w = m_cst.select_child(m_cst.root(), ind + 1);
                int symbol = m_cst.edge(w, 1);
                if (symbol != 1) {
                    pat[0] = symbol;
                    ncomputer(pat, size + 1,lb,rb);
                }
                ++ind;
            }
        } else {
            if (size + 1 <= max_ngram_count) {
                if (freq > 0) {
                    auto node = m_cst.node(lb, rb);
                    int depth = m_cst.depth(node);
                    if (pat.size() == depth) {
                        int ind = 0;
                        while (ind < degree) {
                            auto w = m_cst.select_child(node, ind + 1);
                            int symbol = m_cst.edge(w, depth + 1);
                            if (symbol != 1) {
                                pat.push_back(symbol);
                                ncomputer(pat, size + 1,lb,rb);
                                pat.pop_back();
                            }
                            ++ind;
                        }
                    } else {
                        int symbol = m_cst.edge(node, pat.size() + 1);
                        if (symbol != 1) {
                            pat.push_back(symbol);
                            ncomputer(pat, size + 1,lb,rb);
                            pat.pop_back();
                        }
                    }
                } else {
                }
            }
        }
    }

    index_succinct() = default;
    index_succinct(collection& col)
    {
        std::cout << "CONSTRUCT CST" << std::endl;
        {
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMP";
            cfg.file_map[sdsl::conf::KEm_Y_SA] = col.file_map[KEY_SA];
            cfg.file_map[sdsl::conf::KEm_Y_TEXT_INT] = col.file_map[KEY_TEXT];
            construct(m_cst, col.file_map[KEm_Y_TEXT], cfg, 0);
        }
        std::cout << "DONE" << std::endl;
        std::cout << "CONSTRUCT CST REV" << std::endl;
        {
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMPREV";
            cfg.file_map[sdsl::conf::KEm_Y_SA] = col.file_map[KEY_SAREV];
            cfg.file_map[sdsl::conf::KEm_Y_TEXT_INT] = col.file_map[KEY_TEXTREV];
            construct(m_cst_rev, col.file_map[KEm_Y_TEXTREV], cfg, 0);
        }
        std::cout << "DONE" << std::endl;
        std::cout << "COMPUTE DISCOUNTS" << std::endl;

        m_n1.resize(max_ngram_count + 1);
        m_n2.resize(max_ngram_count + 1);
        m_n3.resize(max_ngram_count + 1);
        m_n4.resize(max_ngram_counti + 1);
        uint64_t lb = 0, rb = m_cst.size() - 1;
        cst_sct3<csa_sada_int<> >::string_type pat(1);
        ncomputer(pat, 0,lb,rb);

        m_Y.resize(max_ngram_count + 1);
        m_D1.resize(max_ngram_count + 1);
        m_D2.resize(max_ngram_count + 1);
        m_D3.resize(max_ngram_count + 1);

        for (int size = 1; size <= max_ngram_count; size++) {
            m_Y[size] = (double)m_n1[size] / (m_n1[size] + 2 * m_n2[size]);
            if (m_n1[size] != 0)
                m_D1[size] = 1 - 2 * m_Y[size] * (double)m_n2[size] / m_n1[size];
            if (m_n2[size] != 0)
                m_D2[size] = 2 - 3 * m_Y[size] * (double)m_n3[size] / m_n2[size];
            if (m_n3[size] != 0)
                m_D3[size] = 3 - 4 * m_Y[size] * (double)m_n4[size] / m_n3[size];
        }
        std::cout << "DONE" << std::endl;
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL, std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_cst.serialize(out, child, "CST");
        written_bytes += m_cst_rev.serialize(out, child, "CST_REV");
        written_bytes += sdsl::serialize_vector(m_N1plus_dotdot);
	written_bytes += sdsl::serialize_vector(m_N3plus_dot);
        written_bytes += sdsl::serialize_vector(m_n1);
        written_bytes += sdsl::serialize_vector(m_n2);
        written_bytes += sdsl::serialize_vector(m_n3);
        written_bytes += sdsl::serialize_vector(m_n4);
        written_bytes += sdsl::serialize_vector(m_Y);
        written_bytes += sdsl::serialize_vector(m_D1);
        written_bytes += sdsl::serialize_vector(m_D2);
        written_bytes += sdsl::serialize_vector(m_D3);
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    void load(std::istream& in)
    {
        m_cst.load(in);
        m_cst_rev.load(in);

        sdsl::load_vector(m_N1plus_dotdot);
        sdsl::load_vector(m_N3plus_dot);

        sdsl::load_vector(m_n1);
        sdsl::load_vector(m_n2);
        sdsl::load_vector(m_n3);
        sdsl::load_vector(m_n4);

        sdsl::load_vector(m_Y);
        sdsl::load_vector(m_D1);
        sdsl::load_vector(m_D2);
        sdsl::load_vector(m_D3);
    }

    void swap(index_succinct& a)
    {
        if (this != &a) {
            m_cst.swap(a.m_cst);
            m_cst_rev.swap(a.m_cst_rev);
            
	    m_N1plus_dotdot.swap(a.m_N1plus_dotdot);
            m_N3plus_dot.swap(a.m_N3plus_dot);
            m_n1.swap(a.m_n1);
            m_n2.swap(a.m_n2);
            m_n3.swap(a.m_n3);
            m_n4.swap(a.m_n4);
            m_Y.swap(a.m_Y);
            m_D1.swap(a.m_D1);
            m_D2.swap(a.m_D2);
            m_D3.swap(a.m_D3);
        }
    }

    uint64_t vocab_size() const
    {
        return m_cst.csa.sigma - 3;
    }
};
