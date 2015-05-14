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
    precomputed_stats(collection&, const t_cst& cst_rev, uint64_t max_ngram_len);

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
    void ncomputer(const t_cst& cst_rev);
    //void ncomputer(const t_cst& cst_rev, uint64_t symbol, int size, uint64_t lb, uint64_t rb);
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
//                        for ... {
// 	            	    switch num_occ {
// 	            		case 2:
// 	            			d.n2[node_depth]++;
// 	            		case 3:
// 	            			d.n3[node_depth]++;
// 	            		case 4:
// 	            			d.n4[node_depth]++;
// 	            	    }
//                        }
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
precomputed_stats::precomputed_stats(collection&, const t_cst& cst_rev, uint64_t max_ngram_len)
    : max_ngram_count(max_ngram_len)
    , N1plus_dotdot(0)
    , N3plus_dot(0)
{
    auto size = max_ngram_count + 1;
    n1.resize(size);
    n2.resize(size);
    n3.resize(size);
    n4.resize(size);
    Y.resize(size);
    Y_cnt.resize(size);
    D1.resize(size);
    D2.resize(size);
    D3.resize(size);
    n1_cnt.resize(size);
    n2_cnt.resize(size);
    n3_cnt.resize(size);
    n4_cnt.resize(size);
    D1_cnt.resize(size);
    D2_cnt.resize(size);
    D3_cnt.resize(size);

    // compute the counts & continuation counts from the CST (reversed)
    ncomputer(cst_rev);

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
precomputed_stats::ncomputer(const t_cst& cst_rev)
{
    for (auto it = cst_rev.begin(); it != cst_rev.end(); ++it) {
        if (it.visit() == 1) {
            auto node = *it;
            auto parent = cst_rev.parent(node);
            auto parent_depth = cst_rev.depth(parent);
            // this next call is expensive for leaves, but we don't care in this case
            // as the for loop below will terminate on the <S> symbol
            auto depth = (!cst_rev.is_leaf(node)) ? cst_rev.depth(node) : (max_ngram_count + 12345);

            auto freq = cst_rev.size(node);
            assert(parent_depth < max_ngram_count);

            for (auto n = parent_depth + 1; n <= std::min(max_ngram_count, depth); ++n) {
                // this call is slow
                auto symbol = cst_rev.edge(node, n);
                // don't count ngrams including these sentinels, including extensions
                if (symbol == EOF_SYM || symbol == EOS_SYM) {
                    it.skip_subtree();
                    break;
                }

                // update frequency counts
                switch (freq) {
                case 1:
                    n1[n] += 1;
                    break;
                case 2:
                    n2[n] += 1;
                    break;
                case 3:
                    n3[n] += 1;
                    break;
                case 4:
                    n4[n] += 1;
                    break;
                }

                if (n == 2)
                    N1plus_dotdot++;
                if (freq >= 3 && n == 1)
                    N3plus_dot++;

                // update continuation counts
                uint64_t n1plus_back = 0ULL;
                if (symbol == PAT_START_SYM)
                    // special case where the pattern starts with <s>: actual count is used
                    n1plus_back = freq;
                else if (n == depth)
                    // no need to adjust for EOS symbol, as this only happens when symbol = <S>
                    n1plus_back = cst_rev.degree(node);
                else
                    n1plus_back = 1;

                switch (n1plus_back) {
                case 1:
                    n1_cnt[n] += 1;
                    break;
                case 2:
                    n2_cnt[n] += 1;
                    break;
                case 3:
                    n3_cnt[n] += 1;
                    break;
                case 4:
                    n4_cnt[n] += 1;
                    break;
                }

                // can skip next evaluations if we know the EOS symbol is coming up next
                if (symbol == PAT_START_SYM) {
                    it.skip_subtree();
                    break;
                }
            }

            if (depth >= max_ngram_count) {
                it.skip_subtree();
            }
        }
    }
}
