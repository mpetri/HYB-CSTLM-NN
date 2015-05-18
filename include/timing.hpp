#pragma once

#include <iostream>
#include <string>
#include <iomanip>
#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>

typedef std::chrono::high_resolution_clock watch;

static std::chrono::nanoseconds n1plusback_time(0);
static std::chrono::nanoseconds n1plusfront_time(0);
static std::chrono::nanoseconds n1plusfrontback_time(0);
static std::chrono::nanoseconds backward_search_time(0);
static std::chrono::nanoseconds forward_search_time(0);
static std::chrono::nanoseconds highestorder_time(0);
static std::chrono::nanoseconds lowerorder_time(0);
static std::chrono::nanoseconds lowestorder_time(0);

//factored out for timing
template <class DS, class Node, class Depth, class Pattern, class Char_Pos>
uint64_t forward_search_X(DS& ds, Node& node, Depth& d, Pattern& p, Char_Pos& char_pos)
{
        auto start = watch::now();
        uint64_t freq= forward_search(ds, node, d, p, char_pos);
        auto end = watch::now();
        forward_search_time +=(end-start);
        return freq;
}

// factored out for timing
template <class DS, class LB1, class RB1, class Pattern, class LB2, class RB2>
uint64_t backward_search_X(DS& ds, LB1 &lb1, RB1 &rb1, Pattern& p, LB2 &lb2, RB2 &rb2)
{
        auto start = watch::now();
        uint64_t freq= backward_search(ds, lb1, rb1, p, lb2, rb2);
        auto end = watch::now();
        backward_search_time +=(end-start);
        return freq;
}


void reset()
{
        n1plusback_time= std::chrono::nanoseconds::zero();
        n1plusfront_time= std::chrono::nanoseconds::zero();
        n1plusfrontback_time= std::chrono::nanoseconds::zero();
        backward_search_time= std::chrono::nanoseconds::zero();
        forward_search_time= std::chrono::nanoseconds::zero();
        highestorder_time= std::chrono::nanoseconds::zero();
        lowerorder_time= std::chrono::nanoseconds::zero();
        lowestorder_time= std::chrono::nanoseconds::zero();
}

void print_timing()
{
     cout << "-----------------TIMING-------------------------" << endl
         <<"N1PLUSBACK ="<< std::chrono::duration_cast<std::chrono::microseconds>(n1plusback_time).count() / 1000.0f << " ms" << endl
         <<"N1PLUSFRONT="<< std::chrono::duration_cast<std::chrono::microseconds>(n1plusfront_time).count() / 1000.0f << " ms" << endl
         <<"N1PLUSFRONTBACK="<< std::chrono::duration_cast<std::chrono::microseconds>(n1plusfrontback_time).count() / 1000.0f << " ms" << endl
         <<"BACKWARDSEARCH="<< std::chrono::duration_cast<std::chrono::microseconds>(backward_search_time).count() / 1000.0f << " ms" << endl
         <<"FORWARDSEARCH="<< std::chrono::duration_cast<std::chrono::microseconds>(forward_search_time).count() / 1000.0f << " ms" << endl
         <<"HIGHESTORDER="<< std::chrono::duration_cast<std::chrono::microseconds>(highestorder_time).count() / 1000.0f << " ms" << endl
         <<"LOWERORDER="<< std::chrono::duration_cast<std::chrono::microseconds>(lowerorder_time).count() / 1000.0f << " ms" << endl
         <<"LOWESTORDER="<< std::chrono::duration_cast<std::chrono::microseconds>(lowestorder_time).count() / 1000.0f << " ms" << endl;
     cout << "------------------------------------------------" << endl;
     cout << "------------------------------------------------" << endl;
     reset();
}
