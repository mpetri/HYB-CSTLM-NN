#pragma once

#include <iostream>

#include "sdsl/vectors.hpp"

#include "constants.hpp"
#include "collection.hpp"
#include "logging.hpp"

template <class t_bv = sdsl::rrr_vector<15>, class t_vec = sdsl::dac_vector<> >
struct compressed_counts {
    typedef sdsl::int_vector<>::size_type size_type;
    typedef t_bv bv_type;
    typedef t_vec vector_type;

private:
    bv_type m_bv;
    typename bv_type::rank_1_type m_bv_rank;
    vector_type m_counts_fb;
    vector_type m_counts_fb1;
    vector_type m_counts_fb2;
    vector_type m_counts_b;
    vector_type m_counts_f1;
    vector_type m_counts_f2;

public:
    compressed_counts() = default;
    compressed_counts(const compressed_counts& cc)
    {
        m_bv = cc.m_bv;
        m_bv_rank.set_vector(&m_bv);
        m_counts_fb = cc.m_counts_fb;
        m_counts_fb1 = cc.m_counts_fb1;
        m_counts_fb2 = cc.m_counts_fb2;
        m_counts_b = cc.m_counts_b;
        m_counts_f1 = cc.m_counts_f1;
        m_counts_f2 = cc.m_counts_f2;
    }
    compressed_counts(compressed_counts&& cc)
    {
        m_bv = std::move(cc.m_bv);
        m_bv_rank.set_vector(&m_bv);
        m_counts_fb = std::move(cc.m_counts_fb);
        m_counts_fb1 = std::move(cc.m_counts_fb1);
        m_counts_fb2 = std::move(cc.m_counts_fb2);
        m_counts_b = std::move(cc.m_counts_b);
        m_counts_f1 = std::move(cc.m_counts_f1);
        m_counts_f2 = std::move(cc.m_counts_f2);
    }
    compressed_counts& operator=(compressed_counts&& cc)
    {
        m_bv = std::move(cc.m_bv);
        m_bv_rank.set_vector(&m_bv);
        m_counts_fb = std::move(cc.m_counts_fb);
        m_counts_fb1 = std::move(cc.m_counts_fb1);
        m_counts_fb2 = std::move(cc.m_counts_fb2);
        m_counts_b = std::move(cc.m_counts_b);
        m_counts_f1 = std::move(cc.m_counts_f1);
        m_counts_f2 = std::move(cc.m_counts_f2);
        return *this;
    }

    template <class t_cst, class t_node_type>
    uint32_t compute_contexts_MKN(t_cst& cst, t_node_type node, uint64_t &num_syms, uint64_t &count1, uint64_t &count2)
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
                    auto size = cs.size(new_node);
                    if (size == 1)
                        count1++;
                    else if (size == 2)
                        count2++;
                } else {
                    auto deg = cst.degree(new_node);
                    total_contexts += deg;
                    // need to know how many of the children have cst.size(new_node) == 1 or 2
#if 1
                    // efficient way to compute, based on earlier pass computing f1 and f2 values
                    uint64_t delta1 = 0, delta2 = 0;
                    lookup_f12(cst, new_node, delta1, delta2); 
                    // FIXME: will all requisite F12 be precomputed, i.e., do they pass under the length limit?
                    count1 += delta1;
                    count2 += delta2;
#else
                    // FIXME: is there a way of doing this more efficiently?
                    //        i'm not sure it's worth it, but we could store the number of c1 & c2 children
                    for (const auto& child : cst.children(new_node)) {
                        auto size = cst.size(child);
                        if (size == 1)
                            count1++;
                        else if (size == 2)
                            count2++;
                    }
#endif
                }
            }
        }
        return total_contexts;
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
                // FIXME: check if cst.size(node(new_lb, new_rb)) == 1 or 2 -- does this make sense?
            } else {
                auto new_node = cst.node(new_lb, new_rb);
                if (cst.is_leaf(new_node) || cst.depth(new_node) != node_depth + 1) {
                    total_contexts++;
                    // FIXME: check if cst.size(new_node) == 1 or 2
                } else {
                    auto deg = cst.degree(new_node);
                    total_contexts += deg;
                    // FIXME: need to know how many of the children have cst.size(new_node) == 1 or 2
                    // (this is going to increase the computational complexity)
                }
            }
        }
        return total_contexts;
    }

    template <class t_cst> compressed_counts(t_cst& cst, uint64_t max_node_depth, bool mkn_counts)
    {
        sdsl::bit_vector tmp_bv(cst.nodes());
        auto tmp_buffer_counts_fb = sdsl::temp_file_buffer<32>::create();
        auto tmp_buffer_counts_fb1 = sdsl::temp_file_buffer<32>::create();
        auto tmp_buffer_counts_fb2 = sdsl::temp_file_buffer<32>::create();
        auto tmp_buffer_counts_b = sdsl::temp_file_buffer<32>::create();
        std::vector<std::pair<uint64_t, uint64_t>> stack;
        auto tmp_buffer_counts_f1 = sdsl::temp_file_buffer<32>::create();
        auto tmp_buffer_counts_f2 = sdsl::temp_file_buffer<32>::create();
        uint64_t num_syms = 0, fb1 = 0, fb2 = 0;

        uint32_t last_node_depth = 0;
        auto root = cst.root();
        for (const auto& child : cst.children(root)) {
            auto itr = cst.begin(child);
            auto end = cst.end(child);

            while (itr != end) {
                auto node = *itr;
                auto depth = cst.node_depth(node);
                //LOG(INFO) << "node id=" << cst.id(node) << " node-depth " << depth << " lb " << cst.lb(node) << " rb " << cst.rb(node);
                //LOG(INFO) << "stack size=" << stack.size() << " values=" << stack;

                if (itr.visit() == 2) {
                    auto node_id = cst.id(node);

                    auto str_depth = cst.depth(node);
                    if (str_depth <= max_node_depth) {
                        tmp_bv[node_id] = 1;

                        if (!mkn_counts) {
                            auto c = compute_contexts(cst, node, num_syms);
                            tmp_buffer_counts_fb.push_back(c);
                            tmp_buffer_counts_b.push_back(num_syms);
                        } else {
                            auto c = compute_contexts(cst, node, num_syms, fb1, fb2);
                            tmp_buffer_counts_fb.push_back(c);
                            tmp_buffer_counts_fb1.push_back(fb1);
                            tmp_buffer_counts_fb2.push_back(fb2);
                            tmp_buffer_counts_b.push_back(num_syms);

                            auto &f12 = stack.back();
                            //LOG(INFO) << "internal node id=" << node_id << " looking up value -> " << f12;
                        
                            tmp_buffer_counts_f1.push_back(f12.first);
                            tmp_buffer_counts_f2.push_back(f12.second);
                            stack.pop_back();
                        }
                    }
                } else {
                    /* first visit */
                    if (! cst.is_leaf(node) ) {
                        auto depth = cst.depth(node);
                        if (depth > max_node_depth) {
                            itr.skip_subtree();
                            //LOG(INFO) << "node is too deep";
                        } 
                    }
                    
                    if (mkn_counts) {
                        int count = cst.size(node);
                        //LOG(INFO) << "node id=" << cst.id(node) << " parent id=" << cst.id(cst.parent(node)) << " count=" << count << " leaf?=" << cst.is_leaf(node);
                        if (depth > last_node_depth)
                            stack.push_back(std::make_pair(0ul, 0ul));
                        auto &cs = stack.back();
                        cs.first += (count == 1);
                        cs.second += (count == 2);
                    }
                }
                ++itr;
                last_node_depth = depth;
            }        
        }
        m_counts_b = vector_type(tmp_buffer_counts_b);
        m_counts_fb = vector_type(tmp_buffer_counts_fb);
        m_counts_fb1 = vector_type(tmp_buffer_counts_fb1);
        m_counts_fb2 = vector_type(tmp_buffer_counts_fb2);
        m_counts_f1 = vector_type(tmp_buffer_counts_f1);
        m_counts_f2 = vector_type(tmp_buffer_counts_f2);

        m_bv = bv_type(tmp_bv);
        m_bv_rank.set_vector(&m_bv);
    }

    // specific MKN implementation, 2-pass
    template <class t_cst> void compressed_counts_MKN(t_cst& cst, uint64_t max_node_depth)
    {
        sdsl::bit_vector tmp_bv(cst.nodes());
        auto tmp_buffer_counts_f1 = sdsl::temp_file_buffer<32>::create();
        auto tmp_buffer_counts_f2 = sdsl::temp_file_buffer<32>::create();
        std::vector<std::pair<uint64_t, uint64_t>> stack;

        // pass 1: initialise f1 and f2 counts
        uint32_t last_node_depth = 0;
        auto root = cst.root();
        for (const auto& child : cst.children(root)) {
            auto itr = cst.begin(child);
            auto end = cst.end(child);

            while (itr != end) {
                auto node = *itr;
                auto depth = cst.node_depth(node);

                if (itr.visit() == 2) {
                    auto node_id = cst.id(node);

                    auto str_depth = cst.depth(node);
                    if (str_depth <= max_node_depth) {
                        tmp_bv[node_id] = 1;
                        auto &f12 = stack.back();
                        
                        tmp_buffer_counts_f1.push_back(f12.first);
                        tmp_buffer_counts_f2.push_back(f12.second);
                        stack.pop_back();
                    }
                } else {
                    /* first visit */
                    if (! cst.is_leaf(node) ) {
                        auto depth = cst.depth(node);
                        if (depth > max_node_depth) {
                            itr.skip_subtree();
                        } 
                    }
                    
                    int count = cst.size(node);
                    if (depth > last_node_depth)
                        stack.push_back(std::make_pair(0ul, 0ul));
                    auto &cs = stack.back();
                    cs.first += (count == 1);
                    cs.second += (count == 2);
                }
                ++itr;
                last_node_depth = depth;
            }        
        }
        // store into compressed in-memory data structures
        m_counts_f1 = vector_type(tmp_buffer_counts_f1);
        m_counts_f2 = vector_type(tmp_buffer_counts_f2);
        m_bv = bv_type(tmp_bv);
        m_bv_rank.set_vector(&m_bv);

        // pass 2: compute front-back (fb, fb1, fb2), back (b) and front (f1, f2) counts
        auto tmp_buffer_counts_fb = sdsl::temp_file_buffer<32>::create();
        auto tmp_buffer_counts_fb1 = sdsl::temp_file_buffer<32>::create();
        auto tmp_buffer_counts_fb2 = sdsl::temp_file_buffer<32>::create();
        auto tmp_buffer_counts_b = sdsl::temp_file_buffer<32>::create();
        uint64_t num_syms = 0;

        auto root = cst.root();
        for (const auto& child : cst.children(root)) {
            auto itr = cst.begin(child);
            auto end = cst.end(child);

            while (itr != end) {
                auto node = *itr;
                auto depth = cst.node_depth(node);

                if (itr.visit() == 2) {
                    auto node_id = cst.id(node);

                    auto str_depth = cst.depth(node);
                    if (str_depth <= max_node_depth) {
                        auto c = compute_contexts(cst, node, num_syms, fb1, fb2);
                        tmp_buffer_counts_fb.push_back(c);
                        tmp_buffer_counts_fb1.push_back(fb1);
                        tmp_buffer_counts_fb2.push_back(fb2);
                        tmp_buffer_counts_b.push_back(num_syms);
                    }
                } else {
                    /* first visit */
                    if (! cst.is_leaf(node) ) {
                        auto depth = cst.depth(node);
                        if (depth > max_node_depth) {
                            itr.skip_subtree();
                        } 
                    }
                }
                ++itr;
            }        
        }
        // compress the count structures into in-memory structures
        m_counts_b = vector_type(tmp_buffer_counts_b);
        m_counts_fb = vector_type(tmp_buffer_counts_fb);
        m_counts_fb1 = vector_type(tmp_buffer_counts_fb1);
        m_counts_fb2 = vector_type(tmp_buffer_counts_fb2);
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
                        std::string name = "") const
    {
        sdsl::structure_tree_node* child
            = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        written_bytes += sdsl::serialize(m_bv, out, child, "bv");
        written_bytes += sdsl::serialize(m_bv_rank, out, child, "bv_rank");
        written_bytes += sdsl::serialize(m_counts_fb, out, child, "counts_fb");
        written_bytes += sdsl::serialize(m_counts_fb1, out, child, "counts_fb1");
        written_bytes += sdsl::serialize(m_counts_fb2, out, child, "counts_fb2");
        written_bytes += sdsl::serialize(m_counts_b, out, child, "counts_b");
        written_bytes += sdsl::serialize(m_counts_f1, out, child, "counts_f1");
        written_bytes += sdsl::serialize(m_counts_f2, out, child, "counts_f2");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    template <class t_cst, class t_node_type>
    bool is_precomputed(t_cst& cst, t_node_type node) const
    {
        auto id = cst.id(node);
        return m_bv[id];
    }
    
    template <class t_cst, class t_node_type>
    void lookup_f12(t_cst& cst, t_node_type node, uint64_t &f1, uint64_t &f2) const
    {
        auto id = cst.id(node);
        auto rank_in_vec = m_bv_rank(id);
        f1 = m_counts_f1[rank_in_vec];
        f2 = m_counts_f2[rank_in_vec];
    }

    template <class t_cst, class t_node_type>
    uint64_t lookup_fb(t_cst& cst, t_node_type node) const
    {
        auto id = cst.id(node);
        auto rank_in_vec = m_bv_rank(id);
        return m_counts_fb[rank_in_vec];
    }

    template <class t_cst, class t_node_type>
    void lookup_fb123p(t_cst& cst, t_node_type node, uint64_t &fb1, uint64_t &fb2, uint64_t &fb3p) const
    {
        auto id = cst.id(node);
        auto rank_in_vec = m_bv_rank(id);
        auto fb = m_counts_fb[rank_in_vec];
        fb1 = m_counts_fb1[rank_in_vec];
        fb2 = m_counts_fb2[rank_in_vec];
        assert(fb >= fb1 + fb2);
        fb3p = fb - fb1 - fb2;
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
        sdsl::load(m_counts_fb1, in);
        sdsl::load(m_counts_fb2, in);
        sdsl::load(m_counts_b, in);
        sdsl::load(m_counts_f1, in);
        sdsl::load(m_counts_f2, in);
    }
};
