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
#include "index_types.hpp"

#include "sample.hpp"

#include "logging.hpp"

using namespace std::chrono;

typedef struct cmdargs {
    std::string pattern_file;
    std::string collection_dir;
    int ngramsize;
    bool ismkn;
    bool isstored;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir> -p <pattern file> -m <boolean> -n <ngramsize>\n",
            program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -p <pattern file>  : the pattern file.\n");
    fprintf(stdout, "  -m : use Modified-KN (default = KN).\n");
    fprintf(stdout, "  -n <ngramsize>  : the ngramsize (integer).\n");
    fprintf(stdout, "  -s : use fastest index lookup using pre-stored counts (implies -b).\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.pattern_file = "";
    args.collection_dir = "";
    args.ismkn = false;
    args.ngramsize = 1;
    args.isstored = false;
    while ((op = getopt(argc, (char* const*)argv, "p:c:n:mbs")) != -1) {
        switch (op) {
        case 'p':
            args.pattern_file = optarg;
            break;
        case 'c':
            args.collection_dir = optarg;
            break;
        case 'm':
            args.ismkn = true;
            break;
        case 'n':
            args.ngramsize = atoi(optarg);
            break;
        case 's':
            args.isstored = true;
            break;
        }
    }
    if (args.collection_dir == "" || args.pattern_file == "") {
        LOG(ERROR) << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

template <class t_idx, class t_rng>
void run_sampling(const t_idx& idx, const std::vector<std::vector<uint64_t> > patterns, uint64_t ngramsize, t_rng &rng)
{
    using clock = std::chrono::high_resolution_clock;
    std::chrono::nanoseconds total_time(0);
    //uint64_t ind = 1;
    lm_bench::reset();
    for (std::vector<uint64_t> pattern : patterns) {
        uint64_t pattern_size = pattern.size();
//        if(pattern.back() ==UNKNOWN_SYM) M--;
        //pattern.push_back(PAT_END_SYM);
        pattern.insert(pattern.begin(), PAT_START_SYM);

        LOG(INFO) << "Pattern is: " << as_string(pattern, idx.m_vocab);
        // run the query
        auto start = clock::now();
        uint64_t next;
        do {
            next = sample_next_symbol(idx, pattern.begin(), pattern.end(), ngramsize, rng);
            pattern.push_back(next);
        } while (next != PAT_END_SYM);
        auto stop = clock::now();

        LOG(INFO) << "Sampled: " << as_string(pattern, idx.m_vocab);
        total_time += (stop - start);
    }
    lm_bench::print();
    LOG(INFO) << "Time = " << duration_cast<microseconds>(total_time).count() / 1000.0f << " ms";
}

int main(int argc, const char* argv[])
{
    log::start_log(argc, argv);

    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);
    assert(!args.ismkn && "not supported yet");

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    /* random number generator */
    std::ranlux24_base rng;
    // FIXME: allow seeding

    index_succinct_compute_n1fb<default_cst_type, default_cst_rev_type> idx;
    index_succinct_store_n1fb<default_cst_type, default_cst_rev_type> idx_store;

    /* load index */
    vocab_uncompressed vocab;
    if (!args.isstored) {

        auto index_file = args.collection_dir + "/index/index-" + sdsl::util::class_to_hash(idx)
                          + ".sdsl";
        if (utils::file_exists(index_file)) {
            LOG(INFO) << "loading index from file '" << index_file << "'";
            sdsl::load_from_file(idx, index_file);
        } else {
            LOG(FATAL) << "index does not exist. build it first";
            return EXIT_FAILURE;
        }

        /* print precomputed parameters */
        idx.print_params(args.ismkn, args.ngramsize);
        vocab = idx.m_vocab;

    } else {

        auto index_file = args.collection_dir + "/index/index-" + sdsl::util::class_to_hash(idx)
                          + ".sdsl";
        if (utils::file_exists(index_file)) {
            LOG(INFO) << "loading index from file '" << index_file << "'";
            sdsl::load_from_file(idx_store, index_file);
        } else {
            LOG(FATAL) << "index does not exist. build it first";
            return EXIT_FAILURE;
        }

        /* print precomputed parameters */
        idx_store.print_params(args.ismkn, args.ngramsize);
        vocab = idx_store.m_vocab;
    }

    /* load patterns */
    std::vector<std::vector<uint64_t> > patterns;
    if (utils::file_exists(args.pattern_file)) {
        std::ifstream ifile(args.pattern_file);
        LOG(INFO) << "reading input file '" << args.pattern_file << "'";
        std::string line;
        while (std::getline(ifile, line)) {
            std::vector<uint64_t> tokens;
            std::istringstream iss(line);
            std::string word;
            while (std::getline(iss, word, ' ')) {
                uint64_t num = vocab.token2id(word, UNKNOWN_SYM);
                tokens.push_back(num);
            }
            patterns.push_back(tokens);
        }
    } else {
        LOG(FATAL) << "cannot read pattern file '" << args.pattern_file << "'";
    }

    /* run sampling */
    if (!args.isstored) {

        run_sampling(idx, patterns, args.ngramsize, rng);

    } else {

        run_sampling(idx_store, patterns, args.ngramsize, rng);
    }

    return EXIT_SUCCESS;
}

std::string as_string(const std::vector<uint64_t> &wids, const vocab_uncompressed &vocab) 
{
    std::ostringstream oss;
    bool first = true;
    for (auto w: wids) {
        if (!first)
            oss << " ";
        oss << vocab.id2token(w);
        first = false;
    }
    return oss.str();
}
