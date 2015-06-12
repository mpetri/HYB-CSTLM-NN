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
    bool isbackward;
    bool isstored;
    bool isreranking;
} cmdargs_t;

std::vector<uint32_t> ngram_occurrences;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir> -p <pattern file> -m <boolean> -n <ngramsize>\n",
            program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -p <pattern file>  : the pattern file.\n");
    fprintf(stdout, "  -m : use Modified-KN (default = KN).\n");
    fprintf(stdout, "  -n <ngramsize>  : the ngramsize (integer).\n");
    fprintf(stdout, "  -b : use faster index lookup using only backward search (default = forward+backward).\n");
    fprintf(stdout, "  -s : use fastest index lookup using pre-stored counts (implies -b).\n");
    fprintf(stdout, "  -r : doing reranking (default = language modelling).\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.pattern_file = "";
    args.collection_dir = "";
    args.ismkn = false;
    args.ngramsize = 1;
    args.isbackward = false;
    args.isstored = false;
    args.isreranking = false;
    while ((op = getopt(argc, (char* const*)argv, "p:c:n:mbsr")) != -1) {
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
        case 'b':
            args.isbackward = true;
            break;
        case 's':
            args.isstored = true;
            args.isbackward = true;
            break;
	case 'r':
	    args.isreranking = true;
        }
    }
    if (args.collection_dir == "" || args.pattern_file == "") {
        LOG(ERROR) << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

// fast_index = true -- use N1+Back/FrontBack based solely on forward CST & backward search
//            = false -- use N1+Back/FrontBack using reverse CST & forward search
template <class t_idx>
void run_queries(const t_idx& idx, const std::vector<std::vector<uint64_t> > patterns,
                 uint64_t ngramsize, bool fast_index, bool ismkn)
{
    using clock = std::chrono::high_resolution_clock;
    double perplexity = 0;
    uint64_t M = 0;
    std::chrono::nanoseconds total_time(0);
    //uint64_t ind = 1;
    lm_bench::reset();
    for (std::vector<uint64_t> pattern : patterns) {
        uint64_t pattern_size = pattern.size();
        std::string pattern_string;
        M += pattern_size + 1; // +1 for adding </s>
//        if(pattern.back() ==UNKNOWN_SYM) M--;
        pattern.push_back(PAT_END_SYM);
        pattern.insert(pattern.begin(), PAT_START_SYM);
        // run the query
        auto start = clock::now();
        double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize, fast_index, ismkn);
        auto stop = clock::now();

        //std::ostringstream sp("", std::ios_base::ate);
        //std::copy(pattern.begin(),pattern.end(),std::ostream_iterator<uint64_t>(sp," "));
        //LOG(INFO) << "P(" << ind++ << ") = " << sp.str() << "("<<
        //duration_cast<microseconds>(stop-start).count() / 1000.0f <<" ms)";

        perplexity += sentenceprob;
        total_time += (stop - start);
    }
    lm_bench::print();
    LOG(INFO) << "Time = " << duration_cast<microseconds>(total_time).count() / 1000.0f << " ms";
    perplexity = perplexity / M;
    LOG(INFO) << "Test Corpus Perplexity is: " << std::setprecision(10) << pow(10, -perplexity);
}

template <class t_idx>
void run_reranker(const t_idx& idx, const std::vector<std::vector<uint64_t> > patterns,
                 uint64_t ngramsize, bool fast_index, bool ismkn)
{
    using clock = std::chrono::high_resolution_clock;
    double perplexity = 0;
    double min = 1000000;
    uint64_t min_candidate_idx = 0;
    uint64_t M = 0;
    std::chrono::nanoseconds total_time(0);
    uint64_t candidate_idx = 1;//line number to find the unconverted sentence
    uint64_t source_idx=0;
    lm_bench::reset();
    for (std::vector<uint64_t> pattern : patterns) {
        if(pattern[0]!=source_idx)
	{
	      std::cout<<"--------------------------------------------------------------"<<endl;
	      std::cout<<" for source_id: "<<source_idx<<
                         " the best candidate is translation number: "<<min_candidate_idx<<
                         " with perplexity: "<<min<<endl;
              std::cout<<"--------------------------------------------------------------"<<endl;
              min= 1000000;
	      min_candidate_idx = 0;
  	}
        source_idx = pattern[0];//stores the source sentence id in n-best submission
	pattern.erase(pattern.begin(), pattern.begin() + 1); //removes sentence_index 
        uint64_t pattern_size = pattern.size();
        std::string pattern_string;
        M = pattern_size + 1; // +1 for adding </s>
        pattern.push_back(PAT_END_SYM);
        pattern.insert(pattern.begin(), PAT_START_SYM);
        // run the query
        auto start = clock::now();
        double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize, fast_index, ismkn);
        auto stop = clock::now();

        //std::ostringstream sp("", std::ios_base::ate);
        //std::copy(pattern.begin(),pattern.end(),std::ostream_iterator<uint64_t>(sp," "));
        //LOG(INFO) << "P(" << ind++ << ") = " << sp.str() << "("<<
        //duration_cast<microseconds>(stop-start).count() / 1000.0f <<" ms)";

        perplexity = pow(10,-sentenceprob/M);
        if(perplexity < min)
	{
		min = perplexity;
		min_candidate_idx = candidate_idx;
	}
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

    /* create collection dir */
    utils::create_directory(args.collection_dir);

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
                uint64_t num = std::stoull(word);
                tokens.push_back(num);
            }
            patterns.push_back(tokens);
        }
    } else {
        LOG(FATAL) << "cannot read pattern file '" << args.pattern_file << "'";
    }

    /* load index */
    if (!args.isbackward && !args.isstored) {
        index_succinct<default_cst_type> idx;
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
        if(args.isreranking)
    	    run_reranker(idx, patterns, args.ngramsize, false, args.ismkn);
	else
            run_queries(idx, patterns, args.ngramsize, false, args.ismkn);

    } else if (args.isbackward && !args.isstored) {

        index_succinct_compute_n1fb<default_cst_type, default_cst_rev_type> idx;
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
    	if(args.isreranking)
            run_reranker(idx, patterns, args.ngramsize, true, args.ismkn);
	else
            run_queries(idx, patterns, args.ngramsize, true, args.ismkn);

    } else if (args.isbackward && args.isstored) {

        index_succinct_store_n1fb<default_cst_type, default_cst_rev_type> idx;
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
        if(args.isreranking)
            run_reranker(idx, patterns, args.ngramsize, true, args.ismkn);
        else
            run_queries(idx, patterns, args.ngramsize, true, args.ismkn);
    }

    LOG(INFO) << "ngram_occs: " << ngram_occurrences;

    return EXIT_SUCCESS;
}
