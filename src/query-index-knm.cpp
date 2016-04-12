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

using namespace cstlm;

using namespace std::chrono;

typedef struct cmdargs {
    std::string pattern_file;
    std::string collection_dir;
    int ngramsize;
    bool ismkn;
    bool isfishy;
    bool isbackward;
    bool isstored;
    bool isreranking;
} cmdargs_t;

std::vector<uint32_t> ngram_occurrences;

void print_usage(const char* program)
{
    fprintf(
        stdout,
        "%s -c <collection dir> -p <pattern file> -m <boolean> -n <ngramsize>\n",
        program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -p <pattern file>  : the pattern file.\n");
    fprintf(stdout, "  -m : use Modified-KN (default = KN).\n");
    fprintf(stdout, "  -n <ngramsize>  : the ngramsize (integer).\n");
    fprintf(stdout, "  -f : use the fishy MKN (default = accurate).\n");
    fprintf(stdout, "  -r : doing reranking (default = language modelling).\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.pattern_file = "";
    args.collection_dir = "";
    args.ismkn = false;
    args.isfishy = false;
    args.ngramsize = 1;
    args.isreranking = false;
    while ((op = getopt(argc, (char* const*)argv, "p:c:n:mfbsr1")) != -1) {
        switch (op) {
        case 'p':
            args.pattern_file = optarg;
            break;
        case 'c':
            args.collection_dir = optarg;
            break;
        case 'f':
            args.isfishy = true;
            break;
        case 'm':
            args.ismkn = true;
            break;
        case 'n':
            args.ngramsize = atoi(optarg);
            break;
        case 'r':
            args.isreranking = true;
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

// fast_index = true -- use N1+Back/FrontBack based solely on forward CST &
// backward search
//            = false -- use N1+Back/FrontBack using reverse CST & forward
//            search
template <class t_idx>
void run_queries(const t_idx& idx,
    const std::vector<typename t_idx::pattern_type> patterns,
    uint64_t ngramsize, bool ismkn, bool isfishy)
{
    using clock = std::chrono::high_resolution_clock;
    double perplexity = 0;
    uint64_t M = 0;
    std::chrono::nanoseconds total_time(0);
    // uint64_t ind = 1;
    lm_bench::reset();
    for (auto pattern : patterns) {
        uint64_t pattern_size = pattern.size();
        std::string pattern_string;
        M += pattern_size + 1; // +1 for adding </s>
        //        if(pattern.back() ==UNKNOWN_SYM) M--;
        pattern.push_back(PAT_END_SYM);
        pattern.insert(pattern.begin(), PAT_START_SYM);
        // run the query
        auto start = clock::now();
        double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize, ismkn, isfishy);
        auto stop = clock::now();

        // std::ostringstream sp("", std::ios_base::ate);
        // std::copy(pattern.begin(),pattern.end(),std::ostream_iterator<uint64_t>(sp,"
        // "));
        // LOG(INFO) << "P(" << ind++ << ") = " << sp.str() << "("<<
        // duration_cast<microseconds>(stop-start).count() / 1000.0f <<" ms)";

        perplexity += sentenceprob;
        total_time += (stop - start);
    }
    lm_bench::print();
    LOG(INFO) << "Time = "
              << duration_cast<microseconds>(total_time).count() / 1000.0f
              << " ms";
    perplexity = perplexity / M;
    LOG(INFO) << "Test Corpus Perplexity is: " << std::setprecision(10)
              << pow(10, -perplexity);
}

template <class t_idx>
void run_reranker(const t_idx& idx,
    const std::vector<typename t_idx::pattern_type> patterns,
    const std::vector<std::vector<std::string> > orig_patterns,
    uint64_t ngramsize, bool ismkn, bool isfishy)
{
    using clock = std::chrono::high_resolution_clock;
    double perplexity = 0;
    double min = 1000000;
    uint64_t best_idx = 0;
    uint64_t M = 0;
    std::chrono::nanoseconds total_time(0);
    // uint64_t candidate_idx = 1;//line number to find the unconverted sentence
    uint64_t source_idx = idx.vocab.token2id("0");
    lm_bench::reset();
    typename t_idx::pattern_type best;
    uint64_t index = 0;
    std::ofstream output;
    output.open("output.rrank");
    for (std::vector<uint64_t> pattern : patterns) {
        if (pattern[0] != source_idx) {
            LOG(INFO) << "Pattern is: "
                      << std::vector<std::string>(orig_patterns[best_idx].begin(),
                             orig_patterns[best_idx].end())
                      << " pplx = " << min;
            std::ostringstream sp("", std::ios_base::ate);
            std::copy(orig_patterns[best_idx].begin(), orig_patterns[best_idx].end(),
                std::ostream_iterator<std::string>(sp, " "));
            output << sp.str() << std::endl;

            min = 1000000;
            best.clear();
            best_idx = 0;
        }
        source_idx = pattern[0]; // stores the source sentence id in n-best submission
        pattern.erase(pattern.begin(),
            pattern.begin() + 2); // removes sentence_index, and |||
        uint64_t pattern_size = pattern.size();
        std::string pattern_string;
        M = pattern_size + 1; // +1 for adding </s>
        pattern.push_back(PAT_END_SYM);
        pattern.insert(pattern.begin(), PAT_START_SYM);
        // run the query
        auto start = clock::now();
        double sentenceprob = sentence_logprob_kneser_ney(idx, pattern, M, ngramsize, ismkn, isfishy);
        auto stop = clock::now();

        perplexity = pow(10, -sentenceprob / M);
        if (perplexity < min) {
            min = perplexity;
            LOG(INFO) << "pplx " << min;
            best_idx = index;
        }
        index++;
        total_time += (stop - start);
    }
    output.close();
    lm_bench::print();
    LOG(INFO) << "Time = "
              << duration_cast<microseconds>(total_time).count() / 1000.0f
              << " ms";
}

std::vector<std::string> parse_line(const std::string& line, alphabet_type alpha)
{
    std::vector<std::string> line_tokens;
    if (alpha == alphabet_type::byte_alphabet) {
        for (const auto& chr : line) {
            line_tokens.push_back(std::string(1, chr));
        }
    }
    else {
        std::istringstream input(line);
        std::string word;
        while (std::getline(input, word, ' ')) {
            line_tokens.push_back(word);
        }
    }
    return line_tokens;
}

template <class t_idx>
int execute(collection& col, const cmdargs_t& args)
{
    /* load index */
    t_idx idx;
    auto index_file = col.path + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if (utils::file_exists(index_file)) {
        LOG(INFO) << "loading index from file '" << index_file << "'";
        sdsl::load_from_file(idx, index_file);
    }
    else {
        LOG(FATAL) << "index " << index_file << " does not exist. build it first";
        return EXIT_FAILURE;
    }

    /* print precomputed parameters */
    idx.print_params(args.ismkn, args.ngramsize);

    /* load patterns */
    std::vector<typename t_idx::pattern_type> patterns;
    std::vector<std::vector<std::string> > orig_patterns;
    if (utils::file_exists(args.pattern_file)) {
        std::ifstream ifile(args.pattern_file);
        LOG(INFO) << "reading input file '" << args.pattern_file << "'";
        std::string line;
        while (std::getline(ifile, line)) {
            auto line_tokens = parse_line(line, col.alphabet);
            typename t_idx::pattern_type tokens;
            std::vector<std::string> orig_tokens;
            for (const auto& token : line_tokens) {
                if (args.isreranking)
                    orig_tokens.push_back(token);
                auto num = idx.vocab.token2id(token, UNKNOWN_SYM);
                tokens.push_back(num);
            }
            if (args.isreranking) {
                orig_tokens.erase(orig_tokens.begin(), orig_tokens.begin() + 2);
                orig_patterns.push_back(orig_tokens);
            }
            patterns.push_back(tokens);
        }
    }
    else {
        LOG(FATAL) << "cannot read pattern file '" << args.pattern_file << "'";
        return EXIT_FAILURE;
    }

    /* run the querying or reranking */
    if (args.isreranking)
        run_reranker(idx, patterns, orig_patterns, args.ngramsize, args.ismkn,
            args.isfishy);
    else
        run_queries(idx, patterns, args.ngramsize, args.ismkn, args.isfishy);

    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    enable_logging = true;
    sdsl::memory_monitor::start();

    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    collection col(args.collection_dir);
    if (col.alphabet == alphabet_type::byte_alphabet) {
        typedef index_succinct<default_cst_byte_type> index_type;
        execute<index_type>(col, args);
    }
    else {
        typedef index_succinct<default_cst_int_type> index_type;
        execute<index_type>(col, args);
    }

    sdsl::memory_monitor::stop();
    LOG(INFO) << "MemoryPeak in query Time " << sdsl::memory_monitor::peak()
              << " bytes.";
}
