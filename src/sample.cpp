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
    fprintf(stdout, "  -s : use fastest index lookup using pre-stored counts.\n");
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
int load_data_and_sample(cmdargs_t &args, t_rng &rng) 
{
    /* load index */
    t_idx idx;
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
    vocab_uncompressed &vocab = idx.m_vocab;

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
    using clock = std::chrono::high_resolution_clock;
    std::chrono::nanoseconds total_time(0);
    lm_bench::reset();
    for (std::vector<uint64_t> pattern : patterns) {
        uint64_t pattern_size = pattern.size();
        //pattern.push_back(PAT_END_SYM);
        pattern.insert(pattern.begin(), PAT_START_SYM);

        LOG(INFO) << "Pattern is: " << as_string(pattern, idx.m_vocab);
        // run the query
        auto start = clock::now();
        uint64_t next;
        do {
            next = sample_next_symbol(idx, pattern.begin(), pattern.end(), args.ngramsize, rng);
            pattern.push_back(next);
        } while (next != PAT_END_SYM);
        auto stop = clock::now();

        LOG(INFO) << "Sampled: " << as_string(pattern, idx.m_vocab);
        total_time += (stop - start);
    }
    lm_bench::print();
    LOG(INFO) << "Time = " << duration_cast<microseconds>(total_time).count() / 1000.0f << " ms";

    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    log::start_log(argc, argv);

    /* types in use */
    typedef index_succinct_compute_n1fb<default_cst_type, default_cst_rev_type> t_idx_compute;
    typedef index_succinct_store_n1fb<default_cst_type, default_cst_rev_type> t_idx_store;
    typedef std::ranlux24_base t_rng;

    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);
    assert(!args.ismkn && "not supported yet");

    /* seed random number generator */
    t_rng rng;
    auto now = std::chrono::system_clock::now();
    rng.seed(std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());

    /* load data, sample */
    if (!args.isstored) {
        return load_data_and_sample<t_idx_compute, t_rng>(args, rng);
    } else {
        return load_data_and_sample<t_idx_store, t_rng>(args, rng);
    }
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
