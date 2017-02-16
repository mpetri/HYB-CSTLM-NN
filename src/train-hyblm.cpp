#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"

#include "index_types.hpp"
#include "logging.hpp"
#include "utils.hpp"

#include "mem_monitor.hpp"

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/gpu-ops.h"
#include "dynet/nodes.h"
#include "dynet/training.h"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "word2vec.hpp"
#include "hyblm.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;
using namespace cstlm;

typedef struct cmdargs {
    std::string collection_dir;
    std::string word2vec_file;
    bool        use_mkn;
} cmdargs_t;


void print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -t <threads>         : limit the number of threads.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int       op;
    args.collection_dir = "";
    args.use_mkn        = true;
    while ((op = getopt(argc, (char* const*)argv, "c:dt:")) != -1) {
        switch (op) {
            case 'c':
                args.collection_dir = optarg;
                break;
            case 't':
                num_cstlm_threads = std::atoi(optarg);
                break;
        }
    }
    if (args.collection_dir == "") {
        LOG(FATAL) << "Missing command line parameters.";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

template <class t_idx>
t_idx load_or_create_cstlm(collection& col, bool use_mkn)
{
    t_idx idx;
    auto  output_file = col.path + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if (cstlm::utils::file_exists(output_file)) {
        LOG(INFO) << "CSTLM loading cstlm index from file : " << output_file;
        std::ifstream ifs(output_file);
        idx.load(ifs);
        return idx;
    }
    using clock = std::chrono::high_resolution_clock;
    auto start  = clock::now();
    idx         = t_idx(col, use_mkn);
    auto stop   = clock::now();
    LOG(INFO) << "CSTLM index construction in (s): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() /
                 1000.0f;

    std::ofstream ofs(output_file);
    if (ofs.is_open()) {
        LOG(INFO) << "CSTLM writing index to file : " << output_file;
        auto bytes = sdsl::serialize(idx, ofs);
        LOG(INFO) << "CSTLM index size : " << bytes / (1024 * 1024) << " MB";
        sdsl::write_structure<sdsl::HTML_FORMAT>(idx, output_file + ".html");
    } else {
        LOG(FATAL) << "CSTLM cannot write index to file : " << output_file;
    }
    return idx;
}

word2vec::embeddings load_or_create_word2vec_embeddings(collection& col)
{
    auto embeddings = word2vec::builder{}
                      .vector_size(300)
                      .window_size(5)
                      .sample_threadhold(1e-5)
                      .num_negative_samples(5)
                      .num_threads(cstlm::num_cstlm_threads)
                      .num_iterations(5)
                      .min_freq_threshold(5)
                      .start_learning_rate(0.025)
                      .train_or_load(col);
    return embeddings;
}

template <class t_cstlm>
hyblm::LM load_or_create_hyblm(collection& col, t_cstlm& cstlm, word2vec::embeddings& WE)
{
    auto hyblm = hyblm::builder{}
                 .dropout(0.3)
                 .layers(2)
                 .vocab_threshold(30000)
                 .hidden_dimensions(128)
                 .sampling(true)
                 .start_learning_rate(0.1)
                 .decay_rate(0.5)
                 .train_or_load(col, cstlm, WE);
    return hyblm;
}


int main(int argc, char** argv)
{
    dynet::initialize(argc, argv);
    enable_logging = true;

    /* parse command line */
    cmdargs_t args = parse_args(argc, (const char**)argv);

    /* (1) parse collection directory and create CSTLM index */
    collection col(args.collection_dir);

    /* (3) create the cstlm model */
    auto cstlm = load_or_create_cstlm<wordlm>(col, args.use_mkn);

    /* (4) load the word2vec embeddings */
    auto word_embeddings = load_or_create_word2vec_embeddings(col);

    /* (5) finally create the hyblm */
    auto hyb_lm = load_or_create_hyblm(col, cstlm, word_embeddings);

    return 0;
}
