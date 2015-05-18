#pragma once

#include "index_succinct.hpp"
#include "index_succinct_store_n1fb.hpp"
#include "index_succinct_compute_n1fb.hpp"

#include "sdsl/suffix_trees.hpp"

using default_csa_type = sdsl::csa_wt_int<>;
using default_cst_type = sdsl::cst_sct3<default_csa_type>;
using default_cst_rev_type = sdsl::cst_sct3<sdsl::csa_sada_int<>>;
