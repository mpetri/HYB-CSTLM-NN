#pragma once

#include "sdsl/rank_support.hpp"
#include "sdsl/suffix_trees.hpp"
#include "index_succinct.hpp"

namespace cstlm {

using default_csa_int_type = sdsl::csa_wt_int<sdsl::wt_huff_int<sdsl::bit_vector, sdsl::rank_support_v<> > >;
using default_cst_int_type = sdsl::cst_sct3<default_csa_int_type,
    sdsl::lcp_dac<>,
    sdsl::bp_support_sada<64, 16, sdsl::rank_support_v<>, sdsl::select_support_mcl<> > >;

using default_csa_byte_type = sdsl::csa_wt<sdsl::wt_huff<sdsl::bit_vector, sdsl::rank_support_v<> > >;
using default_cst_byte_type = sdsl::cst_sct3<default_csa_byte_type,
    sdsl::lcp_dac<>,
    sdsl::bp_support_sada<64, 16, sdsl::rank_support_v<>, sdsl::select_support_mcl<> > >;

using charlm = index_succinct<default_cst_byte_type, 50>;
using wordlm = index_succinct<default_cst_int_type, 10>;
}
