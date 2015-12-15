#pragma once

#include "constants.hpp"
#include "collection.hpp"

#include "sdsl/int_vector_mapper.hpp"

template <uint8_t t_width = 0>
using read_only_mapper = const sdsl::int_vector_mapper<t_width, std::ios_base::in>;

struct precomputed_stats {
    typedef sdsl::int_vector<>::size_type size_type;
    uint64_t max_ngram_count;
    uint64_t N1plus_dotdot;
    uint64_t N3plus_dot;
    uint64_t N1_dot;
    uint64_t N2_dot;
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

    // FIXME: make these class or constructor template arguments
    typedef sdsl::rank_support_v<1> t_rank_bv;
    typedef sdsl::bit_vector::select_1_type t_select_bv;

    precomputed_stats() = default;

    precomputed_stats(collection& col, uint64_t max_ngram_len, bool is_mkn = false);

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
                        std::string name = "") const
    {
        sdsl::structure_tree_node* child
            = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;

        sdsl::write_member(max_ngram_count, out, child, "max_ngram_count");
        sdsl::write_member(N1plus_dotdot, out, child, "N1Plus_dotdot");
        sdsl::write_member(N3plus_dot, out, child, "N3PlusPlus");
        sdsl::write_member(N1_dot, out, child, "N1_dot");
        sdsl::write_member(N2_dot, out, child, "N2_dot");

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
        sdsl::read_member(N1_dot, in);
        sdsl::read_member(N2_dot, in);

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

    template <class t_nums>
    void display_vec(const char* name, const t_nums& nums, size_t ngramsize) const
    {
        LOG(INFO) << name << " = " << t_nums(nums.begin() + 1, nums.begin() + std::min(ngramsize + 1, nums.size()));
    }

    void print(bool ismkn, uint32_t ngramsize) const
    {
        LOG(INFO) << "------------------------------------------------";
        LOG(INFO) << "-------------PRECOMPUTED QUANTITIES-------------";
        LOG(INFO) << "-------------Based on actual counts-------------";

        display_vec("n1", n1, ngramsize);
        display_vec("n2", n2, ngramsize);
        display_vec("n3", n3, ngramsize);
        display_vec("n4", n4, ngramsize);

        LOG(INFO) << "------------------------------------------------";
        display_vec("Y", Y, ngramsize);
        if (ismkn) {
            display_vec("D1", D1, ngramsize);
            display_vec("D2", D2, ngramsize);
            display_vec("D3+", D3, ngramsize);
        }

        LOG(INFO) << "------------------------------------------------";
        LOG(INFO) << "-------------PRECOMPUTED QUANTITIES-------------";
        LOG(INFO) << "-------------Based on continuation counts-------";
        display_vec("N1", n1_cnt, ngramsize);
        display_vec("N2", n2_cnt, ngramsize);
        display_vec("N3", n3_cnt, ngramsize);
        display_vec("N4", n4_cnt, ngramsize);
        LOG(INFO) << "------------------------------------------------";
        display_vec("Yc", Y_cnt, ngramsize);
        if (ismkn) {
            display_vec("D1c", D1_cnt, ngramsize);
            display_vec("D2c", D2_cnt, ngramsize);
            display_vec("D3c", D3_cnt, ngramsize);
        }
        LOG(INFO) << "------------------------------------------------";
        LOG(INFO) << "N1+(..) = " << N1plus_dotdot;
        if (ismkn) {
            LOG(INFO) << "N1(.) = " << N1_dot;
            LOG(INFO) << "N2(.) = " << N2_dot;
            LOG(INFO) << "N3+(.) = " << N3plus_dot;
        }
        LOG(INFO) << "------------------------------------------------";
        LOG(INFO) << "------------------------------------------------";
    }

private:
    template <typename t_cst>
    void ncomputer(collection& col, const t_cst& cst);

    template <class t_cst>
    typename t_cst::size_type
    distance_to_sentinel(read_only_mapper<>& SA,
                         t_rank_bv& sentinel_rank, t_select_bv& sentinel_select,
                         const t_cst& cst, const typename t_cst::node_type& node,
                         const typename t_cst::size_type& offset) const
    {
        auto i = cst.lb(node);
        auto text_offset = SA[i];

        // find count (rank) of 1s in text from [0, offset]
        auto rank = sentinel_rank(text_offset + offset);
        // find the location of the next 1 in the text, this will be </S>
        auto sentinel = sentinel_select(rank + 1);
        return sentinel - text_offset;
    }
};

precomputed_stats::precomputed_stats(collection& col, uint64_t max_ngram_len, bool)
    : max_ngram_count(max_ngram_len)
    , N1plus_dotdot(0)
    , N3plus_dot(0)
    , N1_dot(0)
    , N2_dot(0)

{
    using csa_type = sdsl::csa_wt_int<>;
    using cst_type = sdsl::cst_sct3<csa_type>;
    /* create the reverse CST here as this is the only place we still need it */
    {
        /* create stuff we are missing */
        if (col.file_map.count(KEY_SA) == 0) {
            lm_construct_timer timer(KEY_SA);
            sdsl::int_vector<> sa;
            sdsl::qsufsort::construct_sa(sa, col.file_map[KEY_TEXT].c_str(), 0);
            auto sa_path = col.path + "/" + KEY_PREFIX + KEY_SA;
            sdsl::store_to_file(sa, sa_path);
            col.file_map[KEY_SA] = sa_path;
        }
    }
    cst_type cst;
    {
        auto cst_file = col.path + "/tmp/CST-" + sdsl::util::class_to_hash(cst) + ".sdsl";
        if (!utils::file_exists(cst_file)) {
            lm_construct_timer timer("CST");
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMP";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SA];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXT];
            construct(cst, col.file_map[KEY_TEXT], cfg, 0);
            sdsl::store_to_file(cst, cst_file);
        } else {
            sdsl::load_from_file(cst, cst_file);
        }
    }

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
    ncomputer(col, cst);

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

#if 0
// naive, sloooowwwwww version 
template <class t_cst>
void precomputed_stats::ncomputer(collection& col, const t_cst& cst)
{
    std::vector<typename t_cst::csa_type::value_type> preceding_syms(cst.csa.sigma);
    std::vector<typename t_cst::csa_type::size_type> left(cst.csa.sigma);
    std::vector<typename t_cst::csa_type::size_type> right(cst.csa.sigma);
    uint64_t num_syms;

    for (auto it = cst.begin(); it != cst.end(); ++it) {
        if (it.visit() == 1) {
            auto node = *it;
            auto parent = cst.parent(node);
            // root node
            if (node == parent) continue;
            auto start = cst.edge(node, 1);
            // sentinel subtrees
            if (start == EOF_SYM || start == EOS_SYM) {
                it.skip_subtree();
                continue;
            }
            auto parent_depth = cst.depth(parent);
            auto depth = cst.depth(node);
            auto freq = cst.size(node);
            auto max_n = std::min(max_ngram_count, depth);

            // update continuation counts -- FIXME: this can be done at the root level, i.e., 
            // only update for top level children (with parent_depth == 0)
            uint64_t n1plus_back = 0ULL;
            if (start == PAT_START_SYM)
                // special case where the pattern starts with <s>: actual count is used
                n1plus_back = freq;
            else { 
                // no need to adjust for EOS symbol, as this only happens when start = <S>
                auto lb = cst.lb(node);
                auto rb = cst.rb(node);
                num_syms = 0;
                sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, left, right);
                n1plus_back = num_syms;
            } 

            for (auto n = parent_depth + 1; n <= std::min(max_ngram_count, depth); ++n) {
                auto symbol = cst.edge(node, n);
                if (symbol == EOS_SYM || symbol == EOF_SYM || symbol == UNKNOWN_SYM) {
                    max_n = n-1;
                    it.skip_subtree();
                    break;
                }

                // update frequency counts
                switch (freq) {
                    case 1:
                        n1[n] += 1;
                        if (n == 1) N1_dot++;
                        break;
                    case 2:
                        n2[n] += 1;
                        if (n == 1) N2_dot++;
                        break;
                    case 3: n3[n] += 1; break;
                    case 4: n4[n] += 1; break;
                }
                if (n == 2)
                    N1plus_dotdot++;
                if (freq >= 3 && n == 1)
                    N3plus_dot++;

                switch (n1plus_back) {
                    case 1: n1_cnt[n] += 1; break;
                    case 2: n2_cnt[n] += 1; break;
                    case 3: n3_cnt[n] += 1; break;
                    case 4: n4_cnt[n] += 1; break;
                }
            }

            //std::cerr << "ncomputer -- edge has"
                //<< " range [" << cst.lb(node) << "," << cst.rb(node) << "]" 
                //<< " freq " << freq 
                //<< " n1plus_back " << n1plus_back 
                //<< " depth " << depth
                //<< " pdepth " << parent_depth
                //<< " max_n " << max_n
                //<< " label";
            //for (uint64_t i = 1; i <= std::min(cst.depth(node), max_ngram_count+2); ++i)
                //std::cerr << ' ' << cst.edge(node, i);
            //std::cerr << std::endl;
            //std::cerr << "start is " << start << std::endl;
        }
    }
}

#else

// optimised version
template <class t_cst>
void precomputed_stats::ncomputer(collection& col, const t_cst& cst)
{
    /* load SAREV to speed up edge call */
    read_only_mapper<> SA(col.file_map[KEY_SA]);

    // load up reversed text and store in a bitvector for locating sentinel symbols
    read_only_mapper<> TEXT(col.file_map[KEY_TEXT]);
    sdsl::bit_vector sentinel_bv(TEXT.size());
    for (uint64_t i = 0; i < TEXT.size(); ++i) {
        auto symbol = TEXT[i];
        if (symbol < NUM_SPECIAL_SYMS && symbol != UNKNOWN_SYM && symbol != PAT_START_SYM)
            sentinel_bv[i] = 1;
    }
    t_rank_bv sentinel_rank(&sentinel_bv);
    t_select_bv sentinel_select(&sentinel_bv);

    std::vector<typename t_cst::csa_type::value_type> preceding_syms(cst.csa.sigma);
    std::vector<typename t_cst::csa_type::size_type> left(cst.csa.sigma);
    std::vector<typename t_cst::csa_type::size_type> right(cst.csa.sigma);
    uint64_t num_syms;

    // need to visit subtree corresponding to:
    //      <S> -- PAT_START_SYM    (4)
    //      UNK -- UNKNOWN_SYM      (3)
    //      and all others >= NUM_SPECIAL_SYMS (6)
    
    uint64_t counter = 0; // counter = first symbol on child edge
    for (auto child: cst.children(cst.root())) {
        if (counter != EOF_SYM && counter != EOS_SYM && counter != UNKNOWN_SYM) {
            // process the node
            // FIXME: this can be done more simply by incrementing counter
            //  whenever parent_depth == 0

            for (auto it = cst.begin(child); it != cst.end(child); ++it) {
                if (it.visit() == 1) {
                    auto node = *it;
                    auto parent = cst.parent(node);
                    auto parent_depth = cst.depth(parent);
                    // this next call is expensive for leaves, but we don't care in this case
                    // as the for loop below will terminate on the </S> symbol
                    auto depth = (!cst.is_leaf(node)) ? cst.depth(node) : (max_ngram_count + 12345);
                    auto freq = cst.size(node);
                    assert(parent_depth < max_ngram_count);

                    uint64_t max_n = 0;
                    bool last_is_sentinel = false;
                    if (counter == UNKNOWN_SYM || counter == PAT_END_SYM) {
                        // only need to consider one symbol for UNK, <S>, </S> edges
                        max_n = 1;
                    } else {
                        // need to consider several symbols -- minimum of
                        // 1) edge length; 2) threshold; 3) reaching the </S> token
                        auto distance = distance_to_sentinel(SA, sentinel_rank, sentinel_select, cst, node, parent_depth) + 1;
                        max_n = std::min(max_ngram_count, depth);
                        if (distance <= max_n) {
                            max_n = distance;
                            last_is_sentinel = true;
                        }
                    }

                    // update continuation counts
                    uint64_t n1plus_back = 0ULL;
                    if (counter == PAT_START_SYM) {
                        // special case where the pattern starts with <s>: actual count is used
                        n1plus_back = freq;
                    } else {
                        // no need to adjust for EOS symbol, as this only happens when symbol = <S>
                        auto lb = cst.lb(node);
                        auto rb = cst.rb(node);
                        num_syms = 0;
                        sdsl::interval_symbols(cst.csa.wavelet_tree, lb, rb + 1, num_syms, preceding_syms, left, right);
                        n1plus_back = num_syms;
                    } 

                    //std::cerr << "ncomputer -- edge has"
                        //<< " range [" << cst.lb(node) << "," << cst.rb(node) << "]" 
                        //<< " freq " << freq 
                        //<< " n1plus_back " << n1plus_back 
                        //<< " depth " << cst.depth(node) 
                        //<< " pdepth " << parent_depth
                        //<< " max_n " << max_n
                        //<< " label";
                    //for (uint64_t i = 1; i <= std::min(cst.depth(node), max_ngram_count+2); ++i)
                        //std::cerr << ' ' << cst.edge(node, i);
                    //std::cerr << std::endl;
                    //std::cerr << "counter is " << counter << std::endl;

                    for (auto n = parent_depth + 1; n <= max_n; ++n) {
                        uint64_t symbol = NUM_SPECIAL_SYMS;
                        if (n == 1) symbol = counter;
                        else if (n == max_n && last_is_sentinel) symbol = PAT_END_SYM;

                        // update frequency counts
                        switch (freq) {
                            case 1:
                                n1[n] += 1;
                                if (n == 1) N1_dot++;
                                break;
                            case 2:
                                n2[n] += 1;
                                if (n == 1) N2_dot++;
                                break;
                            case 3: n3[n] += 1; break;
                            case 4: n4[n] += 1; break;
                        }

                        if (n == 2)
                            N1plus_dotdot++;
                        if (freq >= 3 && n == 1)
                            N3plus_dot++;

                        switch (n1plus_back) {
                            case 1: n1_cnt[n] += 1; break;
                            case 2: n2_cnt[n] += 1; break;
                            case 3: n3_cnt[n] += 1; break;
                            case 4: n4_cnt[n] += 1; break;
                        }

                        // can skip subtree if we know the EOS symbol is coming next
                        if (counter == UNKNOWN_SYM || counter == PAT_END_SYM || symbol == PAT_END_SYM) {
                            //std::cerr << "\tquit 1\n";
                            it.skip_subtree();
                            break;
                        }
                    }

                    if (depth >= max_ngram_count) {
                        //std::cerr << "\tquit 2\n";
                        it.skip_subtree();
                    }
                }
            }
        }
        ++counter;
    }
}
#endif
