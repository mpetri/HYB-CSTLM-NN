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

#include "knm.hpp"

#include "logging.hpp"

using namespace std::chrono;

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
        LOG(ERROR) << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

template <class t_idx>
void run_queries(const t_idx& idx, const std::vector<std::vector<uint64_t> > patterns, uint64_t ngramsize)
{
    using clock = std::chrono::high_resolution_clock;
    double perplexity = 0;
    uint64_t M = 0;
    std::chrono::nanoseconds total_time(0);
    uint64_t ind = 1;
    for (std::vector<uint64_t> pattern : patterns) {
        uint64_t pattern_size = pattern.size();
        std::string pattern_string;
        M += pattern_size + 1; // +1 for adding </s>
        pattern.push_back(PAT_END_SYM);
        pattern.insert(pattern.begin(), PAT_START_SYM);
        // run the query
        auto start = clock::now();
        double sentenceprob = run_query_knm(idx, pattern, M, ngramsize);
        auto stop = clock::now();

        std::ostringstream sp("", std::ios_base::ate);
        std::copy(pattern.begin(),pattern.end(),std::ostream_iterator<uint64_t>(sp," "));
        LOG(INFO) << "P(" << ind++ << ") = " << sp.str() << "("<< duration_cast<microseconds>(stop-start).count() / 1000.0f <<" ms)";

        perplexity += sentenceprob;
        total_time += (stop - start);
    }
    LOG(INFO) << "Time = " << duration_cast<microseconds>(total_time).count() / 1000.0f << " ms";
    perplexity = perplexity / M;
    LOG(INFO) << "Test Corpus Perplexity is: " << std::setprecision(10) << pow(10, -perplexity);
}

int main(int argc, const char* argv[])
{
    log::start_log(argc, argv);
    
    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    /* load index */
    index_succinct<default_cst_type> idx;

    auto index_file = args.collection_dir + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if (utils::file_exists(index_file)) {
        LOG(INFO) << "loading index from file '" << index_file << "'";
        sdsl::load_from_file(idx, index_file);
    } else {
        LOG(FATAL) << "index does not exist. build it first";
        return EXIT_FAILURE;
    }

    /* print precomputed parameters */
    idx.print_params(args.ismkn, args.ngramsize);

    /* parse pattern file */
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
                uint64_t num = std::stoull(word);
                tokens.push_back(num);
            }
            patterns.push_back(tokens);
        }
    } else {
        LOG(FATAL) << "cannot read pattern file '" << args.pattern_file << "'";
    }

    run_queries(idx, patterns, args.ngramsize);
    return EXIT_SUCCESS;
}
