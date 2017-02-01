#pragma once

#include "collection.hpp"
#include "query.hpp"

#include <unordered_set>

namespace cstlm {

template <class t_cst, uint32_t t_m, uint32_t t_max_entries>
class prob_cache {
public: // data types
	typedef sdsl::int_vector<>::size_type size_type;
	typedef t_cst						  cst_type;
	typedef typename t_cst::csa_type	  csa_type;
	typedef typename t_cst::node_type	 node_type;
	typedef typename t_cst::string_type   string_type;
	typedef typename csa_type::value_type value_type;
	struct LMQueryMKNCacheData {
		std::vector<node_type> node_incl_vec;
		double				   prob;
	};
	struct LMQueryMKNCacheDataTmp {
		std::vector<value_type> pattern;
		std::vector<node_type>  node_incl_vec;
		double					prob;
		bool operator<(const LMQueryMKNCacheDataTmp& a) const { return prob > a.prob; }
	};
	typedef LMQueryMKNCacheData cache_type;

private:
	std::unordered_map<std::vector<value_type>, LMQueryMKNCacheData> m_cache;
	std::unordered_set<std::vector<value_type>> considered;
	std::priority_queue<LMQueryMKNCacheDataTmp> tmp_cache_pq;

public:
	const uint32_t max_mgram_cache_len	 = t_m;
	const uint32_t max_compute_ngram_len   = 1000;
	const uint32_t max_mgram_cache_entries = t_max_entries;

public:
	prob_cache() = default;
	prob_cache(prob_cache<t_cst, t_m, t_max_entries>&& pc) { m_cache = std::move(pc.m_cache); }
	prob_cache<t_cst, t_m, t_max_entries>& operator=(prob_cache<t_cst, t_m, t_max_entries>&& pc)
	{
		m_cache = std::move(pc.m_cache);
		return (*this);
	}

	template <class t_idx>
	prob_cache(const t_idx& /*idx*/)
	{
		// const auto& cst = idx.cst;
		// uint64_t counter = 0; // counter = first symbol on child edge
		// {
		//     for (auto child : cst.children(cst.root())) {
		//         if(counter != EOF_SYM && counter != EOS_SYM && counter != UNKNOWN_SYM && counter != PAT_END_SYM) {
		//             process_subtree(cst,child,idx);
		//         }
		//         ++counter;
		//     }
		// }
		// considered.clear();
		// while(tmp_cache_pq.size()) {
		//     auto top_entry = tmp_cache_pq.top(); tmp_cache_pq.pop();
		//     LMQueryMKNCacheData d;
		//     d.node_incl_vec = top_entry.node_incl_vec;
		//     d.prob = top_entry.prob;
		//     m_cache[top_entry.pattern] = d;
		// }
	}

	bool add_entry(std::vector<value_type>& pat, std::vector<node_type> niv, double prob)
	{
		if (considered.find(pat) != considered.end()) return true;
		considered.insert(pat);
		bool skip = false;
		if (tmp_cache_pq.size() < max_mgram_cache_entries) {
			LMQueryMKNCacheDataTmp data;
			data.pattern	   = pat;
			data.node_incl_vec = niv;
			data.prob		   = prob;
			tmp_cache_pq.push(data);
		} else {
			double top_prob = tmp_cache_pq.top().prob;
			if (top_prob < prob) {
				tmp_cache_pq.pop();
				LMQueryMKNCacheDataTmp data;
				data.pattern	   = pat;
				data.node_incl_vec = niv;
				data.prob		   = prob;
				tmp_cache_pq.push(data);
			} else {
				skip = true;
			}
		}
		return skip;
	}

	template <class t_idx>
	void process_subtree(const cst_type& cst, const node_type& node, const t_idx& idx)
	{
		auto itr = cst.begin(node);
		auto end = cst.end(node);

		LMQueryMKN<t_idx>			   qmkn(&idx, max_compute_ngram_len, false, false);
		std::vector<value_type>		   pattern;
		std::vector<LMQueryMKN<t_idx>> stack;
		stack.push_back(qmkn);
		while (itr != end) {
			if (itr.visit() == 1) {
				/* get nodes involved */
				auto	 node		  = *itr;
				auto	 parent		  = cst.parent(node);
				auto	 parent_depth = cst.depth(parent);
				uint32_t depth	 = (!cst.is_leaf(node)) ? cst.depth(node) : (max_mgram_cache_len);
				uint32_t add_depth = std::min(depth, max_mgram_cache_len);

				LMQueryMKN<t_idx> cur = stack.back();

				double prob = 0;
				for (size_t i = parent_depth; i < add_depth; i++) {
					uint32_t sym = cst.edge(node, i + 1);
					prob		 = cur.append_symbol_fill_cache(sym, *this);
				}
				bool skip = false;
				if (prob < tmp_cache_pq.top().prob) {
					skip = true;
				}
				if (skip || add_depth >= max_mgram_cache_len) {
					itr.skip_subtree();
				} else {
					stack.push_back(cur);
				}
			} else {
				stack.pop_back();
			}
			++itr;
		}
	}


	size_type serialize(std::ostream&			   out,
						const cst_type&			   cst,
						sdsl::structure_tree_node* v	= NULL,
						std::string				   name = "") const
	{
		sdsl::structure_tree_node* child =
		sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
		size_type written_bytes = 0;

		auto itr = m_cache.begin();
		auto end = m_cache.end();

		sdsl::int_vector<>  pattern_lens(m_cache.size());
		std::vector<double> probabilities(m_cache.size());
		size_t				written_entries = 0;
		size_t				total_plen		= 0;
		while (itr != end) {
			auto pattern = itr->first;
			auto data	= itr->second;
			total_plen += pattern.size();
			pattern_lens[written_entries]  = pattern.size();
			probabilities[written_entries] = data.prob;
			++written_entries;
			++itr;
		}
		sdsl::util::bit_compress(pattern_lens);
		itr = m_cache.begin();
		sdsl::int_vector<> pattern_data(total_plen);
		sdsl::int_vector<> pattern_node_data(total_plen * 2);
		written_entries				= 0;
		size_t node_written_entries = 0;

		while (itr != end) {
			auto pattern = itr->first;
			auto data	= itr->second;

			for (size_t i = 0; i < pattern.size(); i++) {
				pattern_data[written_entries++] = pattern[i];
			}

			for (size_t i = 0; i < pattern.size(); i++) {
				auto lb									  = cst.lb(data.node_incl_vec[i + 1]);
				auto rb									  = cst.rb(data.node_incl_vec[i + 1]);
				pattern_node_data[node_written_entries++] = lb;
				pattern_node_data[node_written_entries++] = rb;
			}
			++itr;
		}
		if (node_written_entries != total_plen * 2) {
			LOG(ERROR) << "node written_entries = " << node_written_entries
					   << " plen*2 = " << total_plen * 2;
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

	void load(std::istream& in, const cst_type& cst)
	{
		sdsl::int_vector<> pattern_lens;
		pattern_lens.load(in);
		std::vector<double> probabilities;
		sdsl::load(probabilities, in);
		sdsl::int_vector<> pattern_data;
		sdsl::load(pattern_data, in);
		sdsl::int_vector<> pattern_node_data;
		sdsl::load(pattern_node_data, in);
		size_t offset = 0;
		auto   root   = cst.root();
		for (size_t i = 0; i < pattern_lens.size(); i++) {
			LMQueryMKNCacheData		d;
			size_t					cur_plen = pattern_lens[i];
			std::vector<value_type> pattern(cur_plen);
			d.prob = probabilities[i];
			d.node_incl_vec.push_back(root);
			for (size_t j = 0; j < cur_plen; j++) {
				pattern[j] = pattern_data[offset + j];
				auto no	= offset + j;
				auto lb	= pattern_node_data[no * 2];
				auto rb	= pattern_node_data[no * 2 + 1];
				auto node  = cst.node(lb, rb);
				d.node_incl_vec.push_back(node);
			}
			m_cache[pattern] = d;
			offset += cur_plen;
		}
	}

	auto find(const std::vector<value_type>& v) const -> decltype(m_cache.find(v))
	{
		return m_cache.find(v);
	}

	auto end() const -> decltype(m_cache.end()) { return m_cache.end(); }
};
}