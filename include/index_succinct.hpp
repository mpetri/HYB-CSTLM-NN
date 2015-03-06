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
    std::vector<double> m_Y;
    std::vector<double> m_D1;
    std::vector<double> m_D2;
    std::vector<double> m_D3;

public:
    index_succinct() = default;
    index_succinct(collection& col)
    {
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
        std::cout << "DONE" << std::endl;
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
        std::cout << "DONE" << std::endl;
        std::cout << "COMPUTE DISCOUNTS" << std::endl;
        // EHSAN TODO

        std::cout << "DONE" << std::endl;
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL, std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += m_cst.serialize(out, child, "CST");
        written_bytes += m_cst_rev.serialize(out, child, "CST_REV");
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
