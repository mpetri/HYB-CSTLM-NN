#pragma once

#include "sdsl/vectors.hpp"

#include "constants.hpp"
#include "collection.hpp"

// traverses over a CST and stores a bit flag for each node in the tree to
// indicate whether or not it contains the sentinel 'EOS' character
// as part of its edge label or ancestor edge label. It also stores the
// offset into the edge label as a delta from the parent depth, with
// special value 0 meaning that it's in an ancestor.
// this allows later calls to avoid this expensive step
// complexity is FIXME

template <class t_bv = sdsl::rrr_vector<15>, class t_vec = sdsl::int_vector<32> >
struct compressed_sentinel_flag {
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_bv bv_type;
    typedef t_vec vector_type;

private:
    bv_type m_bv;
    typename bv_type::rank_1_type m_bv_rank;
    vector_type m_offsets;

public:
    compressed_sentinel_flag() = default;

    template <class t_cst>
    compressed_sentinel_flag(t_cst& cst)
    {
        sdsl::bit_vector has_sentinel(cst.nodes());
        std::map<uint64_t, uint32_t> offsets;

        // use the DFS iterator to traverse `cst`
        for (auto it = cst.begin(); it != cst.end(); ++it) {
            if (it.visit() == 1) {
                // all O(1) ops
                auto node = *it;
                auto node_id = cst.id(node);
                auto parent = cst.parent(node);
                auto parent_id = cst.id(parent);

                if (has_sentinel[parent_id]) {
                    // if a parent contains a sentinel then we just flag this and move on
                    has_sentinel[node_id] = 1;
                    offsets[node_id] = 0;
                }
                else {
                    // check if edge contains sentinel and where
                    auto parent_depth = cst.depth(node);
                    auto depth = cst.depth(node); // can be expensive for leaves, but we
                    // need this (don't we?)
                    auto delta = depth - parent_depth;
                    bool found = false;

                    // pass: the EOS is lexicographical first, so it must be
                    // left-most child when the edge label has only one atom
                    // if (depth == parent_depth + 1 && it != cst.begin(parent)) // pass

                    for (auto d = 1ULL; !found && d <= delta; ++d) {
                        auto l = cst.edge(node, parent_depth + d);
                        if (l == EOS_SYM) {
                            has_sentinel[node_id] = 1;
                            offsets[node_id] = d;
                            found = true;
                        }
                    }

                    if (!found)
                        has_sentinel[node_id] = 0;
                }
            }
        }

        m_bv = bv_type(has_sentinel);
        m_bv_rank.set_vector(&m_bv);
        sdsl::int_vector<32> offset_values(offsets.size());
        size_t i = 0;
        for (auto& node2offset : offsets) {
            offset_values[i] = node2offset.second;
        }
        m_offsets = vector_type(offset_values);
    }

    // returns true/false for whether edge has EOS label
    // offset = 0 -> sentinel is in an ancestor edge
    // offset = 1+ -> sentinel in edge at given offset from parent depth
    template <class t_cst, class t_node_type>
    bool has_sentinel_edge(t_cst& cst, t_node_type node, uint32_t& offset) const
    {
        auto id = cst.id(node);
        if (m_bv[id]) {
            auto rank_in_vec = m_bv_rank(id);
            offset = m_offsets[rank_in_vec];
            return true;
        }
        else
            return false;
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
        std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += sdsl::serialize(m_bv, out, child, "bv");
        written_bytes += sdsl::serialize(m_offsets, out, child, "offsets");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    void load(std::istream& in)
    {
        sdsl::load(m_bv, in);
        m_bv_rank.load(in, &m_bv);
        sdsl::load(m_offsets, in);
    }
};
