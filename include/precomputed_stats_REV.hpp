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
    precomputed_stats(size_t max_ngram)
    {
        max_ngram_count = max_ngram;
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
    }

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
//             	if (parent_depth < max_ngram_count) {
//             		auto stop = std::min(node_depth,max_ngram_count);
//                        for (auto i=parent_depth+1;i<=stop;++i) {
// 	            	switch num_occ {
// 	            		case 2:
// 	            			d.n2[node_depth]++;
// 	            		case 3:
// 	            			d.n3[node_depth]++;
// 	            		case 4:
// 	            			d.n4[node_depth]++;
// 	            	}
//                        }
// 	            	if(parent_depth < 2 && node_depth >= 2) {
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

template <class t_cst>
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

template <class t_cst>
void ncomputer(precomputed_stats& ps, const t_cst& cst_rev, uint64_t symbol, 
        uint64_t size, uint64_t lb, uint64_t rb, uint32_t max_ngram_count, 
        bool check_for_EOS)
{
    auto freq = rb - lb + 1;
    assert(freq >= 1); // otherwise there's a bug
    
    if (size != 0) {
        {
            uint64_t n1plus_back = 0;

            // FIXME: is this the right end of the pattern?
            if (symbol != PAT_START_SYM) {
                auto node = cst_rev.node(lb, rb);

                // depth call expensive for leaves!
                if (size == cst_rev.depth(node)) {
                    n1plus_back = cst_rev.degree(node);
                    // check for EOS
                    if (check_for_EOS && n1plus_back <= 5) {
                        auto w = cst_rev.select_child(node, 1);
                        uint64_t symbol = cst_rev.edge(w, size + 1);
                        if (symbol == EOS_SYM)
                            n1plus_back -= 1;
                    }
                } else {
                    n1plus_back = 1;
                    if (check_for_EOS) {
                        uint64_t symbol = cst_rev.edge(node, size + 1);
                        if (symbol == EOS_SYM)
                            n1plus_back = 0;
                    }
                }
            }
            else //special case where the pattern starts with <s>: acutal count is used
                n1plus_back = freq;
            assert(n1plus_back >= 0);

            switch (n1plus_back) {
                case 1: ps.n1_cnt[size] += 1; break;
                case 2: ps.n2_cnt[size] += 1; break;
                case 3: ps.n3_cnt[size] += 1; break;
                case 4: ps.n4_cnt[size] += 1; break;
                default: /* nothing */; break;
            }
        }

        if (size == 2)
            ps.N1plus_dotdot++;

        switch (freq) {
            case 1: ps.n1[size] += 1; break;
            case 2: ps.n2[size] += 1; break;
            case 3: ps.n3[size] += 1; break;
            case 4: ps.n4[size] += 1; break;
            default: /* nothing */; break;
        }
        
        // MKN needs this
        //if (size == 1)
            //ps.N3plus_dot++;
    }

    if (size == 0) {
        auto v = cst_rev.select_child(cst_rev.root(), 1);
        auto root_id = cst_rev.id(cst_rev.root());
        auto i = 0;
        while (cst_rev.id(v) != root_id) {
            auto child_symbol = 0;
            if (i <= 1) 
                child_symbol = cst_rev.edge(v, 1);
            if ((i > 1) || (child_symbol != EOS_SYM && child_symbol != EOF_SYM)) {
                ncomputer(ps, cst_rev, child_symbol, size + 1, cst_rev.lb(v), cst_rev.rb(v), max_ngram_count, check_for_EOS);
            } else {
                // this is called twice, for the top level sub-trees rooted with 0, 1
                //std::cout << "S1: node " << root_id << " child " << 1 << " symbol " << symbol << "\n";
            }
            v = cst_rev.sibling(v);
            ++i;
        }
    } else {
        if (size + 1 <= max_ngram_count) {
                auto node = cst_rev.node(lb, rb);
                auto depth = cst_rev.depth(node);
                if (size == depth) {
                    auto v = cst_rev.select_child(node, 1);
                    auto root_id = cst_rev.id(cst_rev.root());
                    auto i = 1;

                    while (cst_rev.id(v) != root_id) {
                        // FIXME: this can only happen in first call (due to sort order)
                        auto child_symbol = cst_rev.edge(v, depth + 1);
                        // FIXME: child_symbol never EOS_SYM on undoc data
                        if (symbol != EOS_SYM) {
                            //std::cout << "S2: node " << cst_rev.id(node) << " child " << i
                                      //<< " size " << size << " depth " << depth << " symbol " << child_symbol << "\n";
                            ncomputer(ps, cst_rev, child_symbol, size + 1, cst_rev.lb(v), cst_rev.rb(v), max_ngram_count, check_for_EOS);
                        }
                        v = cst_rev.sibling(v);
                        i += 1;
                    }
                } else {
                    // is the next symbol on the edge a sentinel; if so, stop
                    auto child_symbol = cst_rev.edge(node, size + 1);
                    if (symbol != EOS_SYM) {
                        ncomputer(ps, cst_rev, symbol, size + 1, cst_rev.lb(node), cst_rev.rb(node), max_ngram_count, check_for_EOS);
                    } 
                }
        }
    }
}

template <class t_cst>
precomputed_stats
precompute_statistics(collection&, const t_cst& cst, const t_cst& cst_rev, uint64_t max_ngram_len)
{
    precomputed_stats ps(max_ngram_len);

    ncomputer(ps, cst_rev, 0, 0, 0, cst_rev.size()-1, max_ngram_len, true /* check for EOS */);

    for (auto size = 1ULL; size <= max_ngram_len; size++) {
        ps.Y[size] = ps.n1[size] / (ps.n1[size] + 2 * ps.n2[size]);
        if (ps.n1[size] != 0)
            ps.D1[size] = 1 - 2 * ps.Y[size] * (double)ps.n2[size] / ps.n1[size];
        if (ps.n2[size] != 0)
            ps.D2[size] = 2 - 3 * ps.Y[size] * (double)ps.n3[size] / ps.n2[size];
        if (ps.n3[size] != 0)
            ps.D3[size] = 3 - 4 * ps.Y[size] * (double)ps.n4[size] / ps.n3[size];
    }

    for (auto size = 1ULL; size <= max_ngram_len; size++) {
        ps.Y_cnt[size] = (double)ps.n1_cnt[size] / (ps.n1_cnt[size] + 2 * ps.n2_cnt[size]);
        if (ps.n1_cnt[size] != 0)
            ps.D1_cnt[size] = 1 - 2 * ps.Y_cnt[size] * (double)ps.n2_cnt[size] / ps.n1_cnt[size];
        if (ps.n2_cnt[size] != 0)
            ps.D2_cnt[size] = 2 - 3 * ps.Y_cnt[size] * (double)ps.n3_cnt[size] / ps.n2_cnt[size];
        if (ps.n3_cnt[size] != 0)
            ps.D3_cnt[size] = 3 - 4 * ps.Y_cnt[size] * (double)ps.n4_cnt[size] / ps.n3_cnt[size];
    }

    return ps;
}
