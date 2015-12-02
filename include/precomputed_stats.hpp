#pragma once

#include "constants.hpp"
#include "collection.hpp"

#include "sdsl/int_vector_mapper.hpp"

#include "index_types.hpp"

template <uint8_t t_width = 0>
using read_only_mapper = const sdsl::int_vector_mapper<t_width,std::ios_base::in>;

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

    precomputed_stats(collection& col,uint64_t max_ngram_len, bool is_mkn=false);

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
    void display_vec(const char *name, const t_nums &nums, size_t ngramsize) const
    {
        LOG(INFO) << name << " = " << t_nums(nums.begin()+1, nums.begin() + std::min(ngramsize+1,nums.size()));
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
        if(ismkn){
            LOG(INFO) << "N1(.) = " << N1_dot;
            LOG(INFO) << "N2(.) = " << N2_dot;
            LOG(INFO) << "N3+(.) = " << N3plus_dot;
        }
        LOG(INFO) << "------------------------------------------------";
        LOG(INFO) << "------------------------------------------------";
    }

private:
    template <typename t_cst> void ncomputer(collection& col,const t_cst& cst_rev);


template<class t_cst>
typename t_cst::char_type
emulate_edge(read_only_mapper<>& SAREV,read_only_mapper<>& TREV,const t_cst& cst,
    const typename t_cst::node_type& node,const typename t_cst::size_type& offset)
{
    auto i = cst.lb(node);
    auto text_offset = SAREV[i];
    return TREV[text_offset+offset-1];
}

template<class t_cst>
typename t_cst::size_type
distance_to_sentinel(read_only_mapper<> &SAREV,
        t_rank_bv &sentinel_rank, t_select_bv &sentinel_select,
        const t_cst& cst, const typename t_cst::node_type& node, 
        const typename t_cst::size_type &offset) const
{
    auto i = cst.lb(node);
    auto text_offset = SAREV[i];

    // find count (rank) of 1s in text from [0, offset]
    auto rank = sentinel_rank(text_offset + offset);
    // find the location of the next 1 in the text, this will be the sentence start symbol <S>
    auto sentinel = sentinel_select(rank + 1); 
    return sentinel - text_offset;
}

};

precomputed_stats::precomputed_stats(collection& col,uint64_t max_ngram_len,bool )
    : max_ngram_count(max_ngram_len)
    , N1plus_dotdot(0)
    , N3plus_dot(0)
    , N1_dot(0)
    , N2_dot(0)

{
    /* create the reverse CST here as this is the only place we still need it */
    {
        /* create stuff we are missing */
        if (col.file_map.count(KEY_TEXTREV) == 0) {
            lm_construct_timer timer(KEY_TEXTREV);
            auto textrev_path = col.path + "/" + KEY_PREFIX + KEY_TEXTREV;
            const sdsl::int_vector_mapper<0, std::ios_base::in> sdsl_input(col.file_map[KEY_TEXT]);
            {
                sdsl::int_vector<> tmp;
                std::ofstream ofs(textrev_path);
                sdsl::serialize(tmp, ofs);
            }
            sdsl::int_vector_mapper<0, std::ios_base::out | std::ios_base::in> sdsl_revinput(
                textrev_path);
            sdsl_revinput.resize(sdsl_input.size());
            // don't copy the last two values, sentinels (EOS, EOF)
            std::reverse_copy(std::begin(sdsl_input), std::end(sdsl_input) - 2,
                              std::begin(sdsl_revinput));
            sdsl_revinput[sdsl_input.size() - 2] = EOS_SYM;
            sdsl_revinput[sdsl_input.size() - 1] = EOF_SYM;
            sdsl::util::bit_compress(sdsl_revinput);
            col.file_map[KEY_TEXTREV] = textrev_path;
        }
        
         if (col.file_map.count(KEY_SAREV) == 0) {
            lm_construct_timer timer(KEY_SAREV);
            sdsl::int_vector<> sarev;
            sdsl::qsufsort::construct_sa(sarev, col.file_map[KEY_TEXTREV].c_str(), 0);
            auto sarev_path = col.path + "/" + KEY_PREFIX + KEY_SAREV;
            sdsl::store_to_file(sarev, sarev_path);
            col.file_map[KEY_SAREV] = sarev_path;
         }
    }
    default_cst_type cst_rev;
    {
        auto cst_rev_file = col.path + "/tmp/CST_REV-" + sdsl::util::class_to_hash(cst_rev) + ".sdsl";
        if (!utils::file_exists(cst_rev_file)) {
            lm_construct_timer timer("CST_REV");
            sdsl::cache_config cfg;
            cfg.delete_files = false;
            cfg.dir = col.path + "/tmp/";
            cfg.id = "TMPREV";
            cfg.file_map[sdsl::conf::KEY_SA] = col.file_map[KEY_SAREV];
            cfg.file_map[sdsl::conf::KEY_TEXT_INT] = col.file_map[KEY_TEXTREV];
            construct(cst_rev, col.file_map[KEY_TEXTREV], cfg, 0);
            sdsl::store_to_file(cst_rev, cst_rev_file);
        } else {
            sdsl::load_from_file(cst_rev, cst_rev_file);
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
    ncomputer(col,cst_rev);

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
void precomputed_stats::ncomputer(collection& col,const t_cst& cst_rev)
{
    /* load SAREV to speed up edge call */
    read_only_mapper<> SAREV(col.file_map[KEY_SAREV]);

    // load up reversed text and store in a bitvector for locating sentinel symbols
    read_only_mapper<> TREV(col.file_map[KEY_TEXTREV]);
    sdsl::bit_vector sentinel_bv(TREV.size());
    for (uint64_t i = 0; i < TREV.size(); ++i) {
        auto symbol = TREV[i];
        if (symbol < NUM_SPECIAL_SYMS && symbol != UNKNOWN_SYM)
            sentinel_bv[i] = 1;
    }
    t_rank_bv sentinel_rank(&sentinel_bv);
    t_select_bv sentinel_select(&sentinel_bv);

    /* iterate over all nodes */
    uint64_t counter = 0;
    for (auto it = cst_rev.begin(); it != cst_rev.end(); ++it) {
        if (it.visit() == 1) {
            ++counter;
            // corner cases for counters 1..6, above which are "real" n-grams
            // corresponding to 1 = root; 2 = EOF; 3 = EOS; 4 = UNK; 5 = <S>; 6 = </S>
            // we need to count for 4, 5 & 6; skip subtree for cases 1 -- 5.
            if (counter == 1) {
                continue;
            } else if (counter <= 3) {
                it.skip_subtree();
                continue;
            }

            auto node = *it;
            auto parent = cst_rev.parent(node);
            auto parent_depth = cst_rev.depth(parent);
            // this next call is expensive for leaves, but we don't care in this case
            // as the for loop below will terminate on the <S> symbol
            auto depth = (!cst_rev.is_leaf(node)) ? cst_rev.depth(node) : (max_ngram_count + 12345);
            auto freq = cst_rev.size(node);
            assert(parent_depth < max_ngram_count);

            uint64_t max_n = 0;
            bool last_is_pat_start = false;
            if (4 <= counter && counter <= 6) {
                // only need to consider one symbol for UNK, <S>, </S> edges 
                max_n = 1;
            } else if (counter >= 7) {
                // need to consider several symbols -- minimum of 
                // 1) edge length; 2) threshold; 3) reaching the <S> token
                auto distance = distance_to_sentinel(SAREV,sentinel_rank,sentinel_select,cst_rev,node,parent_depth) + 1;
                max_n = std::min(max_ngram_count, depth);
                if (distance <= max_n) {
                    max_n = distance;
                    last_is_pat_start = true;
                }
            }

            for (auto n = parent_depth + 1; n <= max_n; ++n) {
                uint64_t symbol = NUM_SPECIAL_SYMS;
                if (2 <= counter && counter <= 6) {
                    switch (counter) {
                        //cases 2 & 3 (EOF, EOS) handled above
                        case 4: symbol = UNKNOWN_SYM; break; 
                        case 5: symbol = PAT_START_SYM; break;
                        case 6: symbol = PAT_END_SYM; break; 
                    }
                } else {
                    // edge call is slow, but in these cases all we need to know is if it's <S> or a regular token
                    symbol = (last_is_pat_start && n == max_n) ? PAT_START_SYM : NUM_SPECIAL_SYMS;
                }

                // update frequency counts
                switch (freq) {
                case 1:
                    n1[n] += 1;
                    if(n == 1) N1_dot++; 
                    break;
                case 2:
                    n2[n] += 1;
                    if(n == 1) N2_dot++;
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
                case 1: n1_cnt[n] += 1; break;
                case 2: n2_cnt[n] += 1; break;
                case 3: n3_cnt[n] += 1; break;
                case 4: n4_cnt[n] += 1; break;
                }

                // can skip subtree if we know the EOS symbol is coming next
                if (counter <= 5 || symbol == PAT_START_SYM) { 
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
