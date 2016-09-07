#pragma once

#include "collection.hpp"
#include "query.hpp"

namespace cstlm {

template<class t_cst,uint32_t t_m,uint32_t t_max_entries>
class prob_cache {
public: // data types
    typedef sdsl::int_vector<>::size_type size_type; 
    typedef t_cst cst_type;
    typedef typename t_cst::csa_type csa_type;
    typedef typename t_cst::node_type node_type;
    typedef typename t_cst::string_type string_type;
    typedef typename csa_type::value_type value_type;  
    struct LMQueryMKNCacheData
    {
        std::vector<node_type> node_incl_vec;
        double prob;
    };
    struct LMQueryMKNCacheDataTmp
    {
        std::vector<value_type> pattern;
        std::vector<node_type> node_incl_vec;
        double prob;
        bool operator<(const LMQueryMKNCacheDataTmp& a) const {
            return prob > a.prob;
        }
    };
    typedef LMQueryMKNCacheData cache_type;
private:
    std::unordered_map< std::vector<value_type>, LMQueryMKNCacheData > m_cache;
    const uint32_t max_mgram_cache_len = t_m;
    const uint32_t max_mgram_cache_entries = t_max_entries;
public:
    prob_cache() = default;
    prob_cache(prob_cache<t_cst, t_m, t_max_entries>&& pc)
    {
        m_cache = std::move(pc.m_cache);
    }
    prob_cache<t_cst, t_m,t_max_entries>& operator=(prob_cache<t_cst, t_m,t_max_entries>&& pc)
    {
        m_cache = std::move(pc.m_cache);
        return (*this);
    }
    
    template<class t_idx>
    prob_cache(collection& col, bool is_mkn,const t_idx& idx)
    {
        m_cache;
        const auto& cst = idx.cst;
        uint64_t counter = 0; // counter = first symbol on child edge
        std::priority_queue<LMQueryMKNCacheDataTmp> cache_pq;
        {
            for (auto child : cst.children(cst.root())) {
                if(counter != EOF_SYM && counter != EOS_SYM) {
                    LOG(INFO) << "PROB_CACHE_COMPUTE("<<counter<<")";
                    process_subtree(cst,child,idx,cache_pq);
                }
                ++counter;
            }
        }
        
        while(cache_pq.size()) {
            auto top_entry = cache_pq.top(); cache_pq.pop();
            LMQueryMKNCacheData d;
            d.node_incl_vec = top_entry.node_incl_vec;
            d.prob = top_entry.prob;
            m_cache[top_entry.pattern] = d;
        }
    }
    
    void print_edge_label(const cst_type& cst,const node_type& node) {
        std::vector<uint32_t> label;
        auto depth = (!cst.is_leaf(node)) ? cst.depth(node)
                                                : (max_mgram_cache_len);
        for(size_t i = 0;i<depth;i++) {
            auto sym = cst.edge(node,i+1);
            label.push_back(sym);
        }
        std::string label_str = "'";
        for(size_t i=0;i<label.size()-1;i++) {
            label_str += std::to_string(label[i]) + " ";
        }
        label_str += std::to_string(label.back()) + "'";
        LOG(INFO) << depth << " [" << cst.lb(node) << "," << cst.rb(node) << "] - " << label_str;
    }
    
    template<class t_idx>
    void process_subtree(const cst_type& cst,const node_type& node,const t_idx& idx,
        std::priority_queue<LMQueryMKNCacheDataTmp>& cache_pq) {
        auto itr = cst.begin(node);
        auto end = cst.end(node);
        
        LMQueryMKN<t_idx> qmkn(&idx,max_mgram_cache_len+1, false, false);
        while(itr != end) {
            if (itr.visit() == 1) {
                /* get nodes involved */
                auto node = *itr;
                auto parent = cst.parent(node);
                auto parent_depth = cst.depth(parent);
                uint32_t depth = (!cst.is_leaf(node)) ? cst.depth(node)
                                              : (max_mgram_cache_len);
                uint32_t add_depth = std::min(depth,max_mgram_cache_len);
           
                /* process syms on edge */
                // print_edge_label(cst,node);
                // LOG(INFO) << "parent_depth = " << parent_depth << " depth = " << depth;
                // LOG(INFO) << "pattern size =  " << qmkn.m_pattern.size();
                bool skip = false;
                for(size_t i=parent_depth;i<add_depth;i++) {
                    uint32_t sym = cst.edge(node,i+1);
                    double prob = qmkn.append_symbol(sym,false);
                    
                    LMQueryMKNCacheDataTmp data;
                    data.pattern = std::vector<value_type>(qmkn.m_pattern.begin(), qmkn.m_pattern.end());
                    data.node_incl_vec = qmkn.m_last_nodes_incl;
                    data.prob = prob;
                    
                    if(cache_pq.size() < max_mgram_cache_entries) {
                        cache_pq.push(data);
                    } else {
                        double top_prob = cache_pq.top().prob;
                        if( top_prob < prob ) {
                            cache_pq.pop();
                            cache_pq.push(data);
                        } else {
                            skip = true;
                        }
                    }
                } 
                if (skip || add_depth >= max_mgram_cache_len) {
                    for(size_t i=parent_depth;i<add_depth;i++) {
                        // LOG(INFO) << "POP SYMS " << qmkn.m_pattern.size();
                        qmkn.m_pattern.pop_back();
                        qmkn.m_last_nodes_incl.pop_back();
                    }
                    
                    itr.skip_subtree();
                }
            } else {
                auto node = *itr;
                auto parent = cst.parent(node);
                auto parent_depth = cst.depth(parent);
                uint32_t depth = (!cst.is_leaf(node)) ? cst.depth(node)
                                              : (max_mgram_cache_len);
                uint32_t add_depth = std::min(depth,max_mgram_cache_len);
                
                for(size_t i=parent_depth;i<add_depth;i++) {
                    qmkn.m_pattern.pop_back();
                    qmkn.m_last_nodes_incl.pop_back();
                }
            }
            
            
            ++itr;
        }
    }
    
    
    size_type serialize(std::ostream& out,const cst_type& cst, sdsl::structure_tree_node* v = NULL,std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        
        auto itr = m_cache.begin();
        auto end = m_cache.end();
        
        sdsl::int_vector<> pattern_lens(m_cache.size());
        std::vector<double> probabilities(m_cache.size());
        size_t written_entries = 0;
        size_t total_plen = 0;
        while(itr != end) {
            auto pattern = itr->first;
            auto data = itr->second;
            total_plen += pattern.size();
            pattern_lens[written_entries] = pattern.size();
            probabilities[written_entries] = data.prob;
            ++written_entries;
            ++itr;
        }
        sdsl::util::bit_compress(pattern_lens);
        itr = m_cache.begin();
        sdsl::int_vector<> pattern_data(total_plen);
        sdsl::int_vector<> pattern_node_data(total_plen*2);
        written_entries = 0;
        size_t node_written_entries = 0;
        
        while(itr != end) {
            auto pattern = itr->first;
            auto data = itr->second;

            for(size_t i=0;i<pattern.size();i++) {
                pattern_data[written_entries++] = pattern[i];
            }
            
            for(size_t i=0;i<pattern.size();i++) {
                auto lb = cst.lb(data.node_incl_vec[i+1]);
                auto rb = cst.lb(data.node_incl_vec[i+1]);
                pattern_node_data[node_written_entries++] = lb;
                pattern_node_data[node_written_entries++] = rb;
            }
            
            ++itr;
        }
        sdsl::util::bit_compress(pattern_data);
        sdsl::util::bit_compress(pattern_node_data);
        
        written_bytes += sdsl::serialize(pattern_lens, out, child, "pattern_lens");
        written_bytes += sdsl::serialize(probabilities, out, child, "probabilities");
        written_bytes += sdsl::serialize(pattern_data, out, child, "pattern_data");
        written_bytes += sdsl::serialize(pattern_node_data, out, child, "pattern_node_data");
        
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    void load(std::istream& in,const cst_type& cst)
    {
        sdsl::int_vector<> pattern_lens;
        pattern_lens.load(in);
        std::vector<double> probabilities;
        sdsl::load(probabilities,in);
        sdsl::int_vector<> pattern_data;
        sdsl::load(pattern_data,in);
        sdsl::int_vector<> pattern_node_data;
        sdsl::load(pattern_node_data,in);
        size_t offset = 0;
        auto root = cst.root();
        for(size_t i=0;i<pattern_lens.size();i++) {
            LMQueryMKNCacheData d;
            size_t cur_plen = pattern_lens[i];
            std::vector<value_type> pattern(cur_plen);
            for(size_t j=0;j<cur_plen;j++) pattern[j] = pattern_data[offset + j];
            d.prob = probabilities[i];
            d.node_incl_vec.push_back(root);
            for(size_t j=0;j<cur_plen;j++) {
                auto no = offset+j;
                auto lb = pattern_node_data[no*2];
                auto rb = pattern_node_data[no*2+1];
                auto node = cst.node(lb,rb);
                d.node_incl_vec.push_back(node);
            }
            m_cache[pattern] = d;
            offset += cur_plen;            
        }
        
    }
    
    auto find(const std::vector<value_type>& v) const -> decltype(m_cache.find(v)) {
        return m_cache.find(v);
    }
    
    auto end() const -> decltype(m_cache.end()) {
        return m_cache.end();
    }
};

}