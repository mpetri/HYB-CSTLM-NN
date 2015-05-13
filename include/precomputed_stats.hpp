#pragma once

#include "constants.hpp"
#include "collection.hpp"

struct precomputed_stats {
    typedef sdsl::int_vector<>::size_type size_type;
    uint64_t max_ngram_count;
    double N1plus_dotdot;
    double N3plus_dot;
    std::vector<double> n1;
    std::vector<double> n2;
    std::vector<double> n3;
    std::vector<double> n4;
    std::vector<double> Y;
    std::vector<double> Y_cnt;
    std::vector<double> D1;
    std::vector<double> D2;
    std::vector<double> D3;
    std::vector<double> n1_cnt;
    std::vector<double> n2_cnt;
    std::vector<double> n3_cnt;
    std::vector<double> n4_cnt;
    std::vector<double> D1_cnt;
    std::vector<double> D2_cnt;
    std::vector<double> D3_cnt;

    precomputed_stats() = default;

    template <typename t_cst>
    precomputed_stats(collection&, const t_cst& cst, const t_cst& cst_rev, uint64_t max_ngram_len);

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL, std::string name = "") const
    {
        sdsl::structure_tree_node* child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;

        sdsl::write_member(max_ngram_count, out, child, "max_ngram_count");
        sdsl::write_member(N1plus_dotdot, out, child, "N1PlusPlus");
        sdsl::write_member(N3plus_dot, out, child, "N3PlusPlus");
        sdsl::serialize(n1, out, child, "n1");
        sdsl::serialize(n2, out, child, "n2");
        sdsl::serialize(n3, out, child, "n3");
        sdsl::serialize(n4, out, child, "n4");
        sdsl::serialize(Y, out, child, "Y");
        sdsl::serialize(Y_cnt, out, child, "Y_cnt");
        sdsl::serialize(D1, out, child, "D1");
        sdsl::serialize(D2, out, child, "D2");
        sdsl::serialize(D3, out, child, "D3");
        sdsl::serialize(n1_cnt, out, child, "n1_cnt");
        sdsl::serialize(n2_cnt, out, child, "n2_cnt");
        sdsl::serialize(n3_cnt, out, child, "n3_cnt");
        sdsl::serialize(n4_cnt, out, child, "n4_cnt");

        sdsl::serialize(D1_cnt, out, child, "D1_cnt");
        sdsl::serialize(D2_cnt, out, child, "D2_cnt");
        sdsl::serialize(D3_cnt, out, child, "D3_cnt");

        sdsl::structure_tree::add_size(child, written_bytes);

        return written_bytes;
    }

    void load(std::istream& in)
    {
        sdsl::read_member(max_ngram_count, in);
        sdsl::read_member(N1plus_dotdot, in);
        sdsl::read_member(N3plus_dot, in);

        sdsl::load(n1, in);
        sdsl::load(n2, in);
        sdsl::load(n3, in);
        sdsl::load(n4, in);

        sdsl::load(Y, in);
        sdsl::load(Y_cnt, in);
        sdsl::load(D1, in);
        sdsl::load(D2, in);
        sdsl::load(D3, in);

        sdsl::load(n1_cnt, in);
        sdsl::load(n2_cnt, in);
        sdsl::load(n3_cnt, in);
        sdsl::load(n4_cnt, in);

        sdsl::load(D1_cnt, in);
        sdsl::load(D2_cnt, in);
        sdsl::load(D3_cnt, in);
    }

    void print(bool ismkn, uint32_t ngramsize) const
    {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "-------------PRECOMPUTED QUANTITIES-------------" << std::endl;
        std::cout << "-------------Based on actual counts-------------" << std::endl;
        std::cout << "n1 = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << n1[size] << " ";
        }
        std::cout << std::endl;
        std::cout << "n2 = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << n2[size] << " ";
        }
        std::cout << std::endl;
        std::cout << "n3 = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << n3[size] << " ";
        }
        std::cout << std::endl;
        std::cout << "n4 = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << n4[size] << " ";
        }
        std::cout << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Y = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << Y[size] << " ";
        }
        if (ismkn) {
            std::cout << std::endl;
            std::cout << "D1 = ";
            for (uint32_t size = 0; size <= ngramsize; size++) {
                std::cout << D1[size] << " ";
            }
            std::cout << std::endl;
            std::cout << "D2 = ";
            for (uint32_t size = 0; size <= ngramsize; size++) {
                std::cout << D2[size] << " ";
            }
            std::cout << std::endl;
            std::cout << "D3+= ";
            for (uint32_t size = 0; size <= ngramsize; size++) {
                std::cout << D3[size] << " ";
            }
        }
        std::cout << std::endl;

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "-------------PRECOMPUTED QUANTITIES-------------" << std::endl;
        std::cout << "-------------Based on continuation counts-------" << std::endl;
        std::cout << "n1_cnt = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << n1_cnt[size] << " ";
        }
        std::cout << std::endl;
        std::cout << "n2_cnt = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << n2_cnt[size] << " ";
        }
        std::cout << std::endl;
        std::cout << "n3_cnt = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << n3_cnt[size] << " ";
        }
        std::cout << std::endl;
        std::cout << "n4_cnt = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << n4_cnt[size] << " ";
        }
        std::cout << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Y_cnt = ";
        for (uint32_t size = 0; size <= ngramsize; size++) {
            std::cout << Y_cnt[size] << " ";
        }
        if (ismkn) {
            std::cout << std::endl;
            std::cout << "D1_cnt = ";
            for (uint32_t size = 0; size <= ngramsize; size++) {
                std::cout << D1_cnt[size] << " ";
            }
            std::cout << std::endl;
            std::cout << "D2_cnt = ";
            for (uint32_t size = 0; size <= ngramsize; size++) {
                std::cout << D2_cnt[size] << " ";
            }
            std::cout << std::endl;
            std::cout << "D3+_cnt= ";
            for (uint32_t size = 0; size <= ngramsize; size++) {
                std::cout << D3_cnt[size] << " ";
            }
        }
        std::cout << std::endl;

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "N1+(..) = " << N1plus_dotdot << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

private:
    template <typename t_cst>
    void ncomputer(const t_cst& cst, const t_cst& cst_rev,
                   uint64_t symbol, std::vector<uint64_t> pat,
                   uint64_t lb, uint64_t rb, uint32_t max_ngram_count);
};

// template<class t_cst>
// precomputed_stats
// precompute_statistics(collection& col,const t_cst& cst,uint64_t max_ngram_len) {
// 	precomputed_stats ps(max_ngram_len);

//     // use the DFS iterator to traverse `cst`
//     for (auto it=cst.begin(); it!=cst.end(); ++it) {
//         if (it.visit() == 1) {  // node visited the first time
//             auto v = *it;       // get the node by dereferencing the iterator
//         	auto parent = cst.parent(v);
//         	auto parent_depth = cst.depth(v);
//             auto num_occ = cst.size(v);
//             if(num_occ == 1) {
//             	/* leaf. for all n-grams between parent and MAX -> +1 */
//             	for(size_t i=parent_depth+1;i<=max_ngram_len;i++) {
//             		d.n1[i]++;
//             	}
//             } else {
//             	/* non leaf */
//             	uint64_t node_depth = cst.depth(v);
//             	if(parent_depth < max_ngram_count) {
//             		auto stop = std::min(node_depth,max_ngram_count);

// 	            	switch num_occ {
// 	            		case 2:
// 	            			d.n2[node_depth]++;
// 	            		case 3:
// 	            			d.n3[node_depth]++;
// 	            		case 4:
// 	            			d.n4[node_depth]++;
// 	            	}

// 	            	if(node_depth == 2) {
// 	            		d.N1PlusPlus++;
// 	            	}
// 	            } else {
// 	            	it.skip_subtree();
// 	            }
//             }
//         }
//     }

// 	return ps;
// }

template <typename t_cst>
int N1PlusBack(const t_cst& cst, std::vector<uint64_t> pat, bool check_for_EOS = true)
{
    auto pat_size = pat.size();
    uint64_t n1plus_back = 0;
    uint64_t lb_rev = 0, rb_rev = cst.size() - 1;
    if (backward_search(cst.csa, lb_rev, rb_rev, pat.rbegin(), pat.rend(), lb_rev, rb_rev) > 0) {
        auto node = cst.node(lb_rev, rb_rev);
        if (pat_size == cst.depth(node)) {
            n1plus_back = cst.degree(node);
            if (check_for_EOS) {
                auto w = cst.select_child(node, 1);
                uint64_t symbol = cst.edge(w, pat_size + 1);
                if (symbol == EOS_SYM)
                    n1plus_back = n1plus_back - 1;
            }
        } else {
            if (check_for_EOS) {
                uint64_t symbol = cst.edge(node, pat_size + 1);
                if (symbol != EOS_SYM)
                    n1plus_back = 1;
            } else {
                n1plus_back = 1;
            }
        }
    }
    return n1plus_back;
}

template <typename t_cst>
precomputed_stats::precomputed_stats(collection&, 
        const t_cst& cst, const t_cst& cst_rev, uint64_t max_ngram_len)
{
    max_ngram_count = max_ngram_len;
    auto size = max_ngram_count + 1;
    n1.resize(size); n2.resize(size); n3.resize(size); n4.resize(size);
    Y.resize(size); Y_cnt.resize(size);
    D1.resize(size); D2.resize(size); D3.resize(size);
    n1_cnt.resize(size); n2_cnt.resize(size); n3_cnt.resize(size); n4_cnt.resize(size);
    D1_cnt.resize(size); D2_cnt.resize(size); D3_cnt.resize(size);

    // invoke ncomputer for each subtree
    auto w = cst.select_child(cst.root(), 1);
    auto root_id = cst.id(cst.root());
    std::vector<uint64_t> pat;
    while (cst.id(w) != root_id) {
        auto symbol = cst.edge(w, 1);
        if (symbol != EOS_SYM && symbol != EOF_SYM) {
            ncomputer(cst, cst_rev, symbol, pat,
                    cst.lb(w), cst.rb(w), max_ngram_len);
        }
        w = cst.sibling(w);
    }

    for (auto size = 1ULL; size <= max_ngram_len; size++) {
        Y[size] = n1[size] / (n1[size] + 2 * n2[size]);
        if (n1[size] != 0)
            D1[size] = 1 - 2 * Y[size] * (double)n2[size] / n1[size];
        if (n2[size] != 0)
            D2[size] = 2 - 3 * Y[size] * (double)n3[size] / n2[size];
        if (n3[size] != 0)
            D3[size] = 3 - 4 * Y[size] * (double)n4[size] / n3[size];
    }

    for (auto size = 1ULL; size <= max_ngram_len; size++) {
        Y_cnt[size] = (double)n1_cnt[size] / (n1_cnt[size] + 2 * n2_cnt[size]);
        if (n1_cnt[size] != 0)
            D1_cnt[size] = 1 - 2 * Y_cnt[size] * (double)n2_cnt[size] / n1_cnt[size];
        if (n2_cnt[size] != 0)
            D2_cnt[size] = 2 - 3 * Y_cnt[size] * (double)n3_cnt[size] / n2_cnt[size];
        if (n3_cnt[size] != 0)
            D3_cnt[size] = 3 - 4 * Y_cnt[size] * (double)n4_cnt[size] / n3_cnt[size];
    }
}

template <class t_cst>
void
precomputed_stats::ncomputer(const t_cst& cst, const t_cst& cst_rev,
               uint64_t symbol, std::vector<uint64_t> pat,
               uint64_t lb, uint64_t rb, uint32_t max_ngram_count)
{
    auto freq = rb - lb + 1;
    assert(freq >= 1);
    auto size = pat.size() + 1;

    pat.push_back(symbol);
    uint64_t n1plus_back = 0;

    if (pat[0] != PAT_START_SYM)
        // FIXME: slowwww, better to use (lb, rb) from cst_rev
        n1plus_back = N1PlusBack(cst_rev, pat);
    else 
        // special case where the pattern starts with <s>: actual count is used
        n1plus_back = freq;

    switch (n1plus_back) {
        case 1: n1_cnt[size] += 1; break;
        case 2: n2_cnt[size] += 1; break;
        case 3: n3_cnt[size] += 1; break;
        case 4: n4_cnt[size] += 1; break;
    }

    switch (freq) {
        case 1: n1[size] += 1; break;
        case 2: n2[size] += 1; break;
        case 3: n3[size] += 1; break;
        case 4: n4[size] += 1; break;
    }

    if (size == 2 && freq >= 1) 
        N1plus_dotdot++;

    if (freq >= 3 && size == 1) 
        N3plus_dot++;

    if (size + 1 <= max_ngram_count) {
        auto node = cst.node(lb, rb);
        auto depth = cst.depth(node);
        if (size == depth) {
            // completes an edge
            auto w = cst.select_child(node, 1);
            auto root_id = cst.id(cst.root());

            while (cst.id(w) != root_id) {
                symbol = cst.edge(w, depth + 1);
                if (symbol != EOS_SYM)
                    ncomputer(cst, cst_rev, symbol, pat, cst.lb(w), cst.rb(w), max_ngram_count);
                else
                    assert(false && "this can never happen, EOS is always followed by <S> or EOF");
                w = cst.sibling(w);
            }
        } else {
            // internal to an edge
            symbol = cst.edge(node, size + 1);
            if (symbol != EOS_SYM) 
                ncomputer(cst, cst_rev, symbol, pat, cst.lb(node), cst.rb(node), max_ngram_count);
        }
    }
}
