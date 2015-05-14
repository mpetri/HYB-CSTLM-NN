#pragma once

#include "index_succinct.hpp"
#include "index_succinct_store_n1fb.hpp"

#include "sdsl/suffix_trees.hpp"

using default_csa_type = sdsl::csa_wt_int<>;
using default_cst_type = sdsl::cst_sct3<default_csa_type>;