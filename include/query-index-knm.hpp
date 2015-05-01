#pragma once

#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>
#include <iomanip>
#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"

int ngramsize;
bool ismkn;
const int STARTTAG = 3;
const int ENDTAG = 4;
const int UNKTAG = 2;

typedef struct cmdargs {
    std::string pattern_file;
    std::string collection_dir;
    int ngramsize;
    bool ismkn;
} cmdargs_t;

void
print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir> -p <pattern file> -m <boolean> -n <ngramsize>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -p <pattern file>  : the pattern file.\n");
    fprintf(stdout, "  -m <ismkn>  : the flag for Modified-KN (true), or KN (false).\n");
    fprintf(stdout, "  -n <ngramsize>  : the ngramsize (integer).\n");
};

cmdargs_t
parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.pattern_file = "";
    args.collection_dir = "";
    args.ismkn = false;
    args.ngramsize = 1;
    while ((op = getopt(argc, (char* const*)argv, "p:c:n:m:")) != -1) {
        switch (op) {
        case 'p':
            args.pattern_file = optarg;
            break;
        case 'c':
            args.collection_dir = optarg;
            break;
        case 'm':
            if (strcmp(optarg, "true") == 0)
                args.ismkn = true;
            break;
        case 'n':
            args.ngramsize = atoi(optarg);
            break;
        }
    }
    if (args.collection_dir == "" || args.pattern_file == "") {
        std::cerr << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}


// Computes the probability of P( x | a b c ... ) using raw occurrence counts.
// Note that the backoff probability uses the lower order variants of this method.
//      idx -- the index
//      pattern -- iterators into pattern (is this in order or reversed order???)
//      lb, rb -- left and right bounds on the forward CST (spanning the full index for this method???)
template <class t_idx>
double highestorder(const t_idx& idx, uint64_t level, const bool unk,
                    const std::vector<uint64_t>::iterator& pattern_begin,
                    const std::vector<uint64_t>::iterator& pattern_end,
                    uint64_t& lb, uint64_t& rb,
                    uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
    double backoff_prob = pkn(idx, level, unk,
			      pattern_begin + 1, pattern_end,
                              lb, rb,
                              lb_rev, rb_rev, char_pos, d);
    auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
    uint64_t denominator = 0;
    uint64_t c = 0;

    if (forward_search(idx.m_cst_rev, node, d, *pattern_begin, char_pos) > 0) {
        lb_rev = idx.m_cst_rev.lb(node);
        rb_rev = idx.m_cst_rev.rb(node);
        c = rb_rev - lb_rev + 1;
    }
    int pattern_size = std::distance(pattern_begin, pattern_end);
    double D = 0;
    if(pattern_size == ngramsize)
	D = idx.discount(ngramsize);
    else
	//which is the special case of n<ngramsize that starts with <s>
	D = idx.discount(pattern_size,true);
    double numerator = 0;
    if (!unk && c - D > 0) {
        numerator = c - D;
    }

    uint64_t N1plus_front = 0;
    if (backward_search(idx.m_cst.csa, lb, rb, *pattern_begin, lb, rb) > 0) {
        denominator = rb - lb + 1;
        N1plus_front = idx.N1PlusFront(lb, rb, pattern_begin, pattern_end - 1);
    } else {
		return backoff_prob;
    }

    double output = (numerator / denominator) + (D * N1plus_front / denominator) * backoff_prob;
    return output;
}

template <class t_idx>
double lowerorder(const t_idx& idx, uint64_t level, const bool unk,
                  const std::vector<uint64_t>::iterator& pattern_begin,
                  const std::vector<uint64_t>::iterator& pattern_end,
                  uint64_t& lb, uint64_t& rb,
                  uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
    level = level - 1;
    double backoff_prob = pkn(idx, level, unk,
			      pattern_begin + 1, pattern_end,
                              lb, rb,
                              lb_rev, rb_rev, char_pos, d);

    uint64_t c = 0;
    auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
    int pattern_size = std::distance(pattern_begin, pattern_end);
    if (forward_search(idx.m_cst_rev, node, d, *(pattern_begin), char_pos) > 0) {
        lb_rev = idx.m_cst_rev.lb(node);
        rb_rev = idx.m_cst_rev.rb(node);
        c = idx.N1PlusBack(lb_rev, rb_rev, pattern_size);
    }

    double D = idx.discount(level,true);
    double numerator = 0;
    if (!unk && c - D > 0) {
        numerator = c - D;
    }

    uint64_t N1plus_front = 0;
    uint64_t back_N1plus_front = 0;
    if (backward_search(idx.m_cst.csa, lb, rb, *(pattern_begin), lb, rb) > 0) { //TODO CHECK: what happens to the bounds when this is false?
        back_N1plus_front = idx.N1PlusFrontBack(lb, rb, c, pattern_begin, pattern_end - 1);
        N1plus_front = idx.N1PlusFront(lb, rb, pattern_begin, pattern_end - 1);
	
	if(back_N1plus_front == 0)//TODO check
		// if back_N1plus_front fails to find a full extention to 
		// both left and right, it replaces 0 with extention to right
		// computed by N1plus_front instead. 
		back_N1plus_front = N1plus_front;
     } else {
        return backoff_prob;
    }

    d++;
    double output = (numerator / back_N1plus_front) + (D * N1plus_front / back_N1plus_front) * backoff_prob;
    return output;
}

template <class t_idx>
double lowestorder(const t_idx& idx,
                   const uint64_t& pattern,
                   uint64_t& lb_rev, uint64_t& rb_rev, 
                   uint64_t& char_pos, uint64_t& d)
{
    auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
    double denominator = 0;
    forward_search(idx.m_cst_rev, node, d, pattern, char_pos);
    d++;
    denominator = idx.m_N1plus_dotdot;
    lb_rev = idx.m_cst_rev.lb(node);
    rb_rev = idx.m_cst_rev.rb(node);
    int numerator = idx.N1PlusBack(lb_rev, rb_rev, 1); //TODO precompute this
    double probability = (double)numerator / denominator;
    return probability;
}

//special lowest order handler for P_{KN}(unknown)
template <class t_idx>
double lowestorder_unk(const t_idx& idx)
{
    double denominator = idx.m_N1plus_dotdot;
    double probability = idx.discount(1,true) / denominator;
    return probability;
}


template <class t_idx>
double pkn(const t_idx& idx, uint64_t level, const bool unk,
           const std::vector<uint64_t>::iterator& pattern_begin,
           const std::vector<uint64_t>::iterator& pattern_end,
           uint64_t& lb, uint64_t& rb,
           uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
    int size = std::distance(pattern_begin, pattern_end);
    double probability = 0;
    if ((size == ngramsize && ngramsize != 1) || (*pattern_begin == STARTTAG)) {
        probability = highestorder(idx, level, unk,
				   pattern_begin, pattern_end,
                                   lb, rb,
                                   lb_rev, rb_rev, char_pos, d);
    } else if (size < ngramsize && size != 1) {
        if (size == 0)
            exit(1);

        probability = lowerorder(idx, level, unk,
				 pattern_begin, pattern_end,
                                 lb, rb,
                                 lb_rev, rb_rev, char_pos, d);

    } else if (size == 1 || ngramsize == 1) {
		if(!unk){
	        	probability = lowestorder(idx, *(pattern_end - 1),
	                	                  lb_rev, rb_rev, char_pos, d);
		}else{
		        probability = lowestorder_unk(idx);
		}
	}
    return probability;
}

template <class t_idx>
double run_query_knm(const t_idx& idx, const std::vector<uint64_t>& word_vec, uint64_t& M)
{
    double final_score = 0;
    std::deque<uint64_t> pattern_deq;
    for (const auto& word : word_vec) {
        pattern_deq.push_back(word);
        if (word == STARTTAG)
            continue;
        if (pattern_deq.size() > ngramsize) {
            pattern_deq.pop_front();
        }
        std::vector<uint64_t> pattern(pattern_deq.begin(), pattern_deq.end());
        uint64_t lb_rev = 0, rb_rev = idx.m_cst_rev.size() - 1, lb = 0, rb = idx.m_cst.size() - 1;
        uint64_t char_pos = 0, d = 0;
        int size = std::distance(pattern.begin(), pattern.end());
	bool unk = false;
	if(pattern.back()==77777)
	{
		unk = true;
		M = M - 1;// excluding OOV from perplexity - identical to SRILM ppl
	}
        double score = pkn(idx,size,unk, 
			   pattern.begin(), pattern.end(),
                           lb, rb,
                           lb_rev, rb_rev, char_pos, d);
        final_score += log10(score);
    }
    return final_score;
}


template <class t_idx>
double gate(const t_idx& idx, std::vector<uint64_t> pattern, int nngramsize, bool iismkn)
{
	ngramsize = nngramsize;
	ismkn = iismkn;
	int pattern_size = pattern.size();
	std::string pattern_string;
    pattern.push_back(ENDTAG);
    pattern.insert(pattern.begin(), STARTTAG);
    // run the query
	uint64_t M = pattern_size+1;
    double sentenceprob = run_query_knm(idx, pattern, M);
	double perplexity = pow(10,-(1 / (double) M) * sentenceprob);
	return perplexity;
}
