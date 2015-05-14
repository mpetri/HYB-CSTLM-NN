#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>
#include <iostream>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"

const double d = 0.4;

#include "logging.hpp"

typedef struct cmdargs {
    std::string pattern_file;
    std::string collection_dir;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir> -p <pattern file>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -p <pattern file>  : the pattern file.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.pattern_file = "";
    args.collection_dir = "";
    while ((op = getopt(argc, (char* const*)argv, "p:c:")) != -1) {
        switch (op) {
        case 'p':
            args.pattern_file = optarg;
            break;
        case 'c':
            args.collection_dir = optarg;
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

template <class t_csa> double stupidbackoff(const t_csa& csa_rev, const std::deque<uint64_t>& PRev)
{
    const auto N = csa_rev.size() - 1; // size of the suffix array
    const auto M = PRev.size();

    double score = 0;
    std::vector<uint64_t> full_pattern, context_pattern;
    full_pattern.push_back(PRev[0]);

    // track the lower and upper bounds for numerator (full) and denominator
    // (context) matches
    uint64_t lb_num = 0, lb_denom = 0, rb_num = N - 1, rb_denom = N - 1;
    uint64_t lb_num_prev = 0, lb_denom_prev = 0, rb_num_prev = N - 1, rb_denom_prev = N - 1;
    for (auto m = 1UL; m <= M; m++) {
        double numer = 0, denom = 0;
        lb_num_prev = lb_num;
        rb_num_prev = rb_num;
        sdsl::backward_search(csa_rev, lb_num, rb_num, full_pattern.begin(), full_pattern.end(),
                              lb_num, rb_num);
        numer = rb_num - lb_num + 1;
        // missing patterns || unknown words
        if (lb_num > rb_num || (lb_num < 0 || rb_num > N)) {
            score *= pow(d, std::min(M - m, 5UL));
            // TODO use a smarter backoff weighting
            break;
        }
        rb_denom_prev = rb_denom;
        lb_denom_prev = lb_denom;
        if (m >= 2) {
            sdsl::backward_search(csa_rev, lb_denom, rb_denom, context_pattern.begin(),
                                  context_pattern.end(), lb_denom, rb_denom);
        }
        denom = rb_denom - lb_denom + 1;
        score = numer / denom;
        if (lb_num != rb_num) {
            full_pattern[0] = PRev[m];
        } else {
            // re-use the previous search interval
            lb_num = lb_num_prev;
            rb_num = rb_num_prev;
            // grow the pattern
            full_pattern.push_back(PRev[m]);
        }
        if (lb_denom != rb_denom) {
            if (context_pattern.size() != 0)
                context_pattern[0] = PRev[m];
            else
                context_pattern.push_back(PRev[m]);
        } else {
            // re-use the previous search interval
            lb_denom = lb_denom_prev;
            rb_denom = rb_denom_prev;
            // grow the pattern
            context_pattern.push_back(PRev[m]);
        }
    }

    return score;
}

template <class t_idx>
double run_query_stupid(const t_idx& idx, const std::vector<uint64_t>& word_vec)
{
    double final_score = 1;
    std::deque<uint64_t> pattern;
    for (const auto& word : word_vec) {
        pattern.push_front(word);
        double score = stupidbackoff(idx.m_cst_rev.csa, pattern);
        final_score *= score;
    }
    return final_score;
}

template <class t_idx>
void run_queries(t_idx& idx, const std::string& col_dir,
                 const std::vector<std::vector<uint64_t> > patterns)
{
    using clock = std::chrono::high_resolution_clock;
    auto index_file = col_dir + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if (utils::file_exists(index_file)) {
        LOG(INFO) << "loading index from file '" << index_file << "'";
        sdsl::load_from_file(idx, index_file);

        std::chrono::nanoseconds total_time(0);
        for (const auto& pattern : patterns) {
            // run the query
            auto start = clock::now();
            double score = run_query_stupid(idx, pattern);
            auto stop = clock::now();
            // output score
            std::copy(pattern.begin(), pattern.end(),
                      std::ostream_iterator<uint64_t>(std::cout, " "));
            std::cout << " -> " << score;
            total_time += (stop - start);
        }
        LOG(INFO) << "time = " << duration_cast<microseconds>(total_time).count() / 1000.0f
                  << " ms";
    } else {
        LOG(FATAL) << "index does not exist. build it first";
    }
}

int main(int argc, const char* argv[])
{
    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* create collection dir */
    utils::create_directory(args.collection_dir);

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
        }
    } else {
        LOG(FATAL) << "cannot read pattern file '" << args.pattern_file << "'";
        return EXIT_FAILURE;
    }

    {
        /* load SADA based index */
        using csa_type = sdsl::csa_sada_int<>;
        using cst_type = sdsl::cst_sct3<csa_type>;
        index_succinct<cst_type> idx;

        run_queries(idx, args.collection_dir, patterns);
    }

    {
        /* load WT based index */
        using csa_type = sdsl::csa_wt_int<>;
        using cst_type = sdsl::cst_sct3<csa_type>;
        index_succinct<cst_type> idx;

        run_queries(idx, args.collection_dir, patterns);
    }

    return 0;
}
