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
        index_succinct(collection& col)
        {

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
