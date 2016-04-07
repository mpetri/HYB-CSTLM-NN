#pragma once

#include "sdsl/rank_support.hpp"

using default_csa_type = sdsl::csa_wt_int<sdsl::wt_huff_int<sdsl::bit_vector, sdsl::rank_support_v<> > >;
using default_cst_type = sdsl::cst_sct3<default_csa_type,
    sdsl::lcp_dac<>,
    sdsl::bp_support_sada<64, 16, sdsl::rank_support_v<>, sdsl::select_support_mcl<> > >;

#include "index_succinct.hpp"
#include "sdsl/suffix_trees.hpp"
