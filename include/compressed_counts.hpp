#pragma once

#include "sdsl/vectors.hpp"

#include "constants.hpp"
#include "collection.hpp"

template <class t_bv = sdsl::rrr_vector<15>, class t_vec = sdsl::int_vector<32> >
struct compressed_counts {
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_bv bv_type;
    typedef t_vec vector_type;

private:
    bv_type m_bv;
    typename bv_type::rank_1_type m_bv_rank;
    vector_type m_counts_fb;
    vector_type m_counts_b;

public:
    compressed_counts() = default;
    compressed_counts(const compressed_counts& cc)
    {
        m_bv = cc.m_bv;
        m_bv_rank.set_vector(&m_bv);
        m_counts_fb = cc.m_counts_fb;
        m_counts_b = cc.m_counts_b;
    }
    compressed_counts(compressed_counts&& cc)
    {
        m_bv = std::move(cc.m_bv);
        m_bv_rank.set_vector(&m_bv);
        m_counts_fb = std::move(cc.m_counts_fb);
        m_counts_b = std::move(cc.m_counts_b);
    }
    compressed_counts& operator=(compressed_counts&& cc)
    {
        m_bv = std::move(cc.m_bv);
        m_bv_rank.set_vector(&m_bv);
        m_counts_fb = std::move(cc.m_counts_fb);
        m_counts_b = std::move(cc.m_counts_b);
        return *this;
    }

    template <class t_cst, class t_node_type>
    uint32_t compute_contexts(t_cst& cst, t_node_type node, uint64_t &num_syms)
    {
        static std::vector<typename t_cst::csa_type::value_type> preceding_syms(cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::size_type> left(cst.csa.sigma);
        static std::vector<typename t_cst::csa_type::size_type> right(cst.csa.sigma);
        auto lb = cst.lb(node);
        auto rb = cst.rb(node);
        num_syms = 0;
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
                auto new_node_depth = cst.depth(new_node);
                if (new_node_depth != node_depth + 1) {
                    total_contexts++;
                } else {
                    auto deg = cst.degree(new_node);
                    total_contexts += deg;
                }
            }
        }
        return total_contexts;
    }

    template <class t_cst> compressed_counts(t_cst& cst, uint64_t max_node_depth)
    {
        sdsl::bit_vector tmp_bv(cst.nodes());
        std::map<uint64_t, uint32_t> counts_fb, counts_b;
        uint64_t num_syms = 0;

        auto root = cst.root();
        for (const auto& child : cst.children(root)) {
            auto itr = cst.begin(child);
            auto end = cst.end(child);
            while (itr != end) {
                if (itr.visit() == 1) {
                    auto node = *itr;
                    auto node_id = cst.id(node);
                    if (cst.is_leaf(node)) {
                        tmp_bv[node_id] = 0;
                    } else {
                        auto depth = cst.depth(node);
                        if (depth > max_node_depth) {
                            // FIXME: want to store counts_b values for these too
                            // but not sure it's worth having all the trivial
                            // n1+fb values stored alongside.
                            tmp_bv[node_id] = 0;
                            itr.skip_subtree();
                        } else {
                            auto c = compute_contexts(cst, node, num_syms);
                            counts_fb[node_id] = c;
                            counts_b[node_id] = num_syms;
                            tmp_bv[node_id] = 1;
                        }
                    }
                }
                ++itr;
            }
        }

        sdsl::int_vector<32> cnts(counts_fb.size());
        auto itr = counts_fb.begin();
        auto end = counts_fb.end();
        auto citr = cnts.begin();
        while (itr != end) {
            *citr = itr->second;
            ++citr;
            ++itr;
        }
        m_counts_fb = vector_type(cnts);
        
        cnts = sdsl::int_vector<32>(counts_b.size());
        itr = counts_b.begin();
        end = counts_b.end();
        citr = cnts.begin();
        while (itr != end) {
            *citr = itr->second;
            ++citr;
            ++itr;
        }
        m_counts_b = vector_type(cnts);

        m_bv = bv_type(tmp_bv);
        m_bv_rank.set_vector(&m_bv);
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
                        std::string name = "") const
    {
        sdsl::structure_tree_node* child
            = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += sdsl::serialize(m_bv, out, child, "bv");
        written_bytes += sdsl::serialize(m_counts_fb, out, child, "counts_fb");
        written_bytes += sdsl::serialize(m_counts_b, out, child, "counts_b");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    template <class t_cst, class t_node_type> 
    uint64_t lookup_fb(t_cst& cst, t_node_type node) const
    {
        auto id = cst.id(node);
        auto rank_in_vec = m_bv_rank(id);
        return m_counts_fb[rank_in_vec];
    }

    template <class t_cst, class t_node_type> 
    uint64_t lookup_b(t_cst& cst, t_node_type node) const
    {
        auto id = cst.id(node);
        auto rank_in_vec = m_bv_rank(id);
        return m_counts_b[rank_in_vec];
    }

    void load(std::istream& in)
    {
        sdsl::load(m_bv, in);
        m_bv_rank.load(in, &m_bv);
        sdsl::load(m_counts_fb, in);
        sdsl::load(m_counts_b, in);
    }
};
