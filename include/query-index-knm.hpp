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

// computes N_1+( * abc ) equivalent to computing N_1+ ( cba *) in the reverse suffix tree
template <class t_idx>
int N1PlusBack(const t_idx& idx, const uint64_t& lb_rev, const uint64_t& rb_rev, int patrev_size, bool check_for_EOS = true)
{
	uint64_t c = 0;
	auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
	if (patrev_size == idx.m_cst_rev.depth(node)) {
	    c = idx.m_cst_rev.degree(node);
	    if (check_for_EOS) {
	        auto w = idx.m_cst_rev.select_child(node, 1);
	        uint64_t symbol = idx.m_cst_rev.edge(w, patrev_size + 1);
	        if (symbol == 1)
	            c = c - 1;
	    }
	} else {
	    if (check_for_EOS) {
	        uint64_t symbol = idx.m_cst_rev.edge(node, patrev_size + 1);
	        if (symbol != 1)
	            c = 1;
	    } else {
	        c = 1;
	    }
	}
	return c;
}

template <class t_idx>
double discount(const t_idx& idx, int level, bool cnt=false)
{
	if(cnt)
	return idx.m_Y_cnt[level];
	else
	return idx.m_Y[level];
}

//  Computes N_1+( * ab * )
//  n1plus_front = value of N1+( * abc ) (for some following symbol 'c')
//  if this is N_1+( * ab ) = 1 then we know the only following symbol is 'c'
//  and thus N1+( * ab * ) is the same as N1+( * abc ), stored in n1plus_back
template <class t_idx>
uint64_t N1PlusFrontBack(const t_idx& idx,
	                     const uint64_t& lb, const uint64_t& rb,
	                     const uint64_t n1plus_back,
	                     const std::vector<uint64_t>::iterator& pattern_begin,
	                     const std::vector<uint64_t>::iterator& pattern_end,
	                     bool check_for_EOS = true)
{
	// ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
	// ASSUMPTION: pattern_begin, pattern_end cover just the pattern we're interested in (i.e., we want N1+ dot pattern dot)
	int pattern_size = std::distance(pattern_begin, pattern_end);
	auto node = idx.m_cst.node(lb, rb);
	uint64_t back_N1plus_front = 0;
	uint64_t lb_rev_prime = 0, rb_rev_prime = idx.m_cst_rev.size() - 1;
	uint64_t lb_rev_stored = 0, rb_rev_stored = 0;
	// this is a full search for the pattern in reverse order in the reverse tree!
	for (auto it = pattern_begin; it != pattern_end and lb_rev_prime <= rb_rev_prime;) {
	    backward_search(idx.m_cst_rev.csa,
	                    lb_rev_prime, rb_rev_prime,
	                    *it,
	                    lb_rev_prime, rb_rev_prime);
	    it++;
	}
	// this is when the pattern matches a full edge in the CST
	if (pattern_size == idx.m_cst.depth(node)) {
	    auto w = idx.m_cst.select_child(node, 1);
	    int root_id = idx.m_cst.id(idx.m_cst.root());
	    while (idx.m_cst.id(w) != root_id) {
	        lb_rev_stored = lb_rev_prime;
	        rb_rev_stored = rb_rev_prime;
	        uint64_t symbol = idx.m_cst.edge(w, pattern_size + 1);
	        if (symbol != 1 || !check_for_EOS) {
	            // find the symbol to the right
	            // (which is first in the reverse order)
	            backward_search(idx.m_cst_rev.csa,
	                            lb_rev_stored, rb_rev_stored,
	                            symbol,
	                            lb_rev_stored, rb_rev_stored);

	            back_N1plus_front += N1PlusBack(idx, lb_rev_stored, rb_rev_stored, pattern_size + 1, check_for_EOS);
	        }
	        w = idx.m_cst.sibling(w);
	    }
	    return back_N1plus_front;
	} else {
	    // special case, only one way of extending this pattern to the right
	    return n1plus_back;
	}
}

// Computes N_1+( abc * )
template <class t_idx>
uint64_t N1PlusFront(const t_idx& idx,
	                 const uint64_t& lb, const uint64_t& rb,
	                 std::vector<uint64_t>::iterator pattern_begin,
	                 std::vector<uint64_t>::iterator pattern_end,
	                 bool check_for_EOS = true)
{
	// ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
	auto node = idx.m_cst.node(lb, rb);
	int pattern_size = std::distance(pattern_begin, pattern_end);
	uint64_t N1plus_front = 0;
	if (pattern_size == idx.m_cst.depth(node)) {
	    auto w = idx.m_cst.select_child(node, 1);
	    N1plus_front = idx.m_cst.degree(node);
	    if (check_for_EOS) {
	        uint64_t symbol = idx.m_cst.edge(w, pattern_size + 1);
	        if (symbol == 1) {
	            N1plus_front = N1plus_front - 1;
	        }
	    }
	    return N1plus_front;
	} else {
	    if (check_for_EOS) {
	        uint64_t symbol = idx.m_cst.edge(node, pattern_size + 1);
	        if (symbol != 1) {
	            N1plus_front = 1;
	        }
	    }
	    return N1plus_front;
	}
}

// Computes the probability of P( x | a b c ... ) using raw occurrence counts.
// Note that the backoff probability uses the lower order variants of this method.
//      idx -- the index
//      pattern -- iterators into pattern (is this in order or reversed order???)
//      lb, rb -- left and right bounds on the forward CST (spanning the full index for this method???)
template <class t_idx>
double highestorder(const t_idx& idx, uint64_t level,
	                const std::vector<uint64_t>::iterator& pattern_begin,
	                const std::vector<uint64_t>::iterator& pattern_end,
	                uint64_t& lb, uint64_t& rb,
	                uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
	double backoff_prob = pkn(idx, level, 
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
	D = discount(idx,ngramsize);
	else
	//which is the special case of n<ngramsize that starts with <s>
	D = discount(idx,pattern_size,true);
	double numerator = 0;
	if (c - D > 0) {
	    numerator = c - D;
	}

	uint64_t N1plus_front = 0;
	if (backward_search(idx.m_cst.csa, lb, rb, *pattern_begin, lb, rb) > 0) {
	    denominator = rb - lb + 1;
	    N1plus_front = N1PlusFront(idx, lb, rb, pattern_begin, pattern_end - 1);
	} else {
	return backoff_prob;
	}

	double output = (numerator / denominator) + (D * N1plus_front / denominator) * backoff_prob;
	return output;
}
 
template <class t_idx>
double lowerorder(const t_idx& idx, uint64_t level,
	              const std::vector<uint64_t>::iterator& pattern_begin,
	              const std::vector<uint64_t>::iterator& pattern_end,
	              uint64_t& lb, uint64_t& rb,
	              uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
	level = level - 1;
	double backoff_prob = pkn(idx, level, 
				  pattern_begin + 1, pattern_end,
	                          lb, rb,
	                          lb_rev, rb_rev, char_pos, d);

	uint64_t c = 0;
	auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
	int pattern_size = std::distance(pattern_begin, pattern_end);
	if (forward_search(idx.m_cst_rev, node, d, *(pattern_begin), char_pos) > 0) {
	    lb_rev = idx.m_cst_rev.lb(node);
	    rb_rev = idx.m_cst_rev.rb(node);
	    c = N1PlusBack(idx, lb_rev, rb_rev, pattern_size);
	}

	double D = discount(idx,level,true);
	double numerator = 0;
	if (c - D > 0) {
	    numerator = c - D;
	}

	uint64_t N1plus_front = 0;
	uint64_t back_N1plus_front = 0;
	if (backward_search(idx.m_cst.csa, lb, rb, *(pattern_begin), lb, rb) > 0) { //TODO CHECK: what happens to the bounds when this is false?
	    back_N1plus_front = N1PlusFrontBack(idx, lb, rb, c, pattern_begin, pattern_end - 1);
	    N1plus_front = N1PlusFront(idx, lb, rb, pattern_begin, pattern_end - 1);

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
	               uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
	auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
	double denominator = 0;

	forward_search(idx.m_cst_rev, node, d, pattern, char_pos);
	d++;
	denominator = idx.m_N1plus_dotdot;
	lb_rev = idx.m_cst_rev.lb(node);
	rb_rev = idx.m_cst_rev.rb(node);
	int numerator = N1PlusBack(idx, lb_rev, rb_rev, 1); //TODO precompute this
	double probability = (double)numerator / denominator;
	return probability;
}

template <class t_idx>
double pkn(const t_idx& idx, uint64_t level,
	       const std::vector<uint64_t>::iterator& pattern_begin,
	       const std::vector<uint64_t>::iterator& pattern_end,
	       uint64_t& lb, uint64_t& rb,
	       uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
	int size = std::distance(pattern_begin, pattern_end);
	double probability = 0;
	if ((size == ngramsize && ngramsize != 1) || (*pattern_begin == STARTTAG)) {
	   
	    probability = highestorder(idx, level,
				   pattern_begin, pattern_end,
	                               lb, rb,
	                               lb_rev, rb_rev, char_pos, d);

	} else if (size < ngramsize && size != 1) {
	    if (size == 0)
	        exit(1);

	    probability = lowerorder(idx, level,
				 pattern_begin, pattern_end,
	                             lb, rb,
	                             lb_rev, rb_rev, char_pos, d);

	} else if (size == 1 || ngramsize == 1) {
	    probability = lowestorder(idx, *(pattern_end - 1),
	                              lb_rev, rb_rev, char_pos, d);
	}
	return probability;
}

template <class t_idx>
double run_query_knm(const t_idx& idx, const std::vector<uint64_t>& word_vec)
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
	    double score = pkn(idx,size, 
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
    double sentenceprob = run_query_knm(idx, pattern);
	double perplexity = pow(10,-(1 / (double) (pattern_size+1 )) * sentenceprob);
	return perplexity;
}
