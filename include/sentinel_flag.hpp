#pragma once

#include "sdsl/vectors.hpp"

#include "constants.hpp"
#include "collection.hpp"

// traverses over a CST and stores a bit flag for each node in the tree to indicate whether
// or not it contains the sentinel 'EOS' character as part of its edge label
// this allows later calls to avoid this expensive step
// complexity is FIXME

struct compressed_sentinel_flag 
{
    typedef sdsl::int_vector<>::size_type size_type;
    typedef sdsl::rrr_vector<63> bv_type;

private:
    bv_type m_bv;
    bv_type::rank_1_type m_bv_rank;

public:

    compressed_sentinel_flag() = default;

    template<class t_cst>
    compressed_sentinel_flag(t_cst& cst)
    {
        sdsl::bit_vector has_sentinel(cst.nodes());

        // use the DFS iterator to traverse `cst`
        for (auto it=cst.begin(); it!=cst.end(); ++it) 
        {
            if (it.visit() == 1) 
            {   // node visited the first time
                auto node = *it;       // get the node by dereferencing the iterator           
                auto node_id = cst.id(node);
                auto parent = cst.parent(node);
                auto parent_depth = cst.depth(node);

                // check if edge contains sentinel
                auto depth = cst.depth(node); // can be expensive for leaves
                bool seen = false;
                for (auto d = parent_depth+1; !seen && d <= depth; ++d) 
                {
                    auto l = cst.edge(node, d);
                    if (l == EOS_SYM) seen = true;
                }
                has_sentinel[node_id] = seen;
                // FIXME: could also compute n1..4, N1..4 in this pass
            }
        }

        m_bv = bv_type(has_sentinel);
    }

    template<class t_cst,class t_node_type>
    bool has_sentinel_edge(t_cst& cst,t_node_type node) const 
    {
    	auto id = cst.id(node);
    	return m_bv[id];
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL, std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += sdsl::serialize(m_bv,out,child,"bv");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    void load(std::istream& in)
    {
        sdsl::load(m_bv, in);
        m_bv_rank.load(in,&m_bv);
    }
};
