#pragma once

#include "utils.hpp"

#include <sdsl/suffix_arrays.hpp>


template<class t_cst>
class index_succinct
{
    public:
        typedef sdsl::int_vector<>::size_type               size_type;
    protected:
        t_cst  m_cst;
        t_cst  m_cst_rev;
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
                construct(m_cst,col.file_map[KEY_TEXT],cfg,0);
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
                construct(m_cst_rev,col.file_map[KEY_TEXTREV],cfg,0);
            }
            std::cout << "DONE" << std::endl;
        }

        size_type serialize(std::ostream& out,sdsl::structure_tree_node* v=NULL, std::string name="")const
        {
            sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
            size_type written_bytes = 0;
            written_bytes += m_cst.serialize(out, child, "CST");
            written_bytes += m_cst_rev.serialize(out, child, "CST_REV");
            sdsl::structure_tree::add_size(child, written_bytes);
            return written_bytes;
        }

        void load(std::istream& in)
        {
            m_csa.load(in);
            m_cst_rev.load(in);
        }

        void swap(index_succinct& a)
        {
            if (this != &a) {
                m_cst.swap(a.m_cst);
                m_cst_rev.swap(a.m_cst_rev);
            }
        }

};
