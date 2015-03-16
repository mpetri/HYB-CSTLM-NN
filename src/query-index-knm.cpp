#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <string>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"

int N = 0, N1 = 0, N2 = 0, N3 = 0, denominator = 0;
int ngramsize;
bool ismkn;
uint64_t STARTTAG = 3;
uint64_t ENDTAG = 4;
int freq = 0;
uint64_t dot_LB = -5, dot_RB = -5;
uint64_t dot_LB_dot = -5, dot_RB_dot = -5;

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
        std::cerr << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

template <class t_idx>
void N1PlusFrontBack_Back(const t_idx& idx, std::vector<uint64_t> pat, uint64_t lb, uint64_t rb)
{
    auto v = idx.m_cst_rev.node(lb, rb);
    int size = pat.size();
    freq = rb - lb + 1;
    if (freq == 1 && lb != rb) {
        freq = 0;
    }
    if (freq > 0) {
        if (size == idx.m_cst_rev.depth(v)) {
            auto w = idx.m_cst_rev.select_child(v, 1);
            int symbol = idx.m_cst_rev.edge(w, size + 1);
            denominator = idx.m_cst_rev.degree(v);
            if (symbol == 1) {
                denominator = denominator - 1;
            }
        } else {
            int symbol = idx.m_cst_rev.edge(v, size + 1);
            if (symbol != 1) {
                denominator = 1;
            }
        }
    }
}

template <class t_idx>
void N1PlusFrontBack_Front(const t_idx& idx, std::vector<uint64_t> pat, uint64_t lb, uint64_t rb)
{
    auto v = idx.m_cst.node(lb, rb);
    freq = rb - lb + 1;
    if (freq == 1 && lb != rb) {
        freq = 0;
    }
    int size = pat.size();
    if (freq > 0) {
        if (size == idx.m_cst.depth(v)) {
            auto w = idx.m_cst.select_child(v, 1);
            int root_id = idx.m_cst.id(idx.m_cst.root());
            while (idx.m_cst.id(w) != root_id) {
                int symbol = idx.m_cst.edge(w, size + 1);
                if (symbol != 1) {
                    N++;
                    pat.push_back(symbol);
                    uint64_t lbrev = idx.m_cst_rev.lb(w), rbrev = idx.m_cst_rev.rb(w);
                    backward_search(idx.m_cst_rev.csa, lbrev, rbrev, pat.rbegin(), pat.rend(), lbrev, rbrev);
                    std::vector<uint64_t> patrev = pat; //TODO replace this
                    reverse(patrev.begin(), patrev.end());
                    N1PlusFrontBack_Back(idx, patrev, lbrev, rbrev);
                    pat.pop_back();
                }
                w = idx.m_cst.sibling(w);
            }
        } else {
            uint64_t lbrev = 0, rbrev = idx.m_cst_rev.size() - 1;
            backward_search(idx.m_cst_rev.csa, lbrev, rbrev, pat.rbegin(), pat.rend(), lbrev, rbrev);
            N1PlusFrontBack_Back(idx, pat, lbrev, rbrev);
        }
    }
}

template <class t_idx>
int discount(const t_idx& idx, int c)
{
    double D = 0;
    if (ismkn) {
        if (c == 1) {
            if (idx.m_n1[ngramsize] != 0)
                D = idx.m_D1[ngramsize];
        } else if (c == 2) {
            if (idx.m_n2[ngramsize] != 0)
                D = idx.m_D2[ngramsize];
        } else if (c >= 3) {
            if (idx.m_n3[ngramsize] != 0)
                D = idx.m_D3[ngramsize];
        }
    } else {
        D = idx.m_Y[ngramsize];
    }
    return D;
}

template <class t_idx>
void N1PlusFront(const t_idx& idx, uint64_t lb, uint64_t rb, std::vector<uint64_t> pat)
{
    auto node = idx.m_cst.node(lb, rb);
    int pat_size = pat.size();
    int deg = idx.m_cst.degree(node);
    if (pat_size == idx.m_cst.depth(node)) {
        if (!ismkn) {
            auto w = idx.m_cst.select_child(node, 1);
            int symbol = idx.m_cst.edge(w, pat_size + 1);
            N = deg;
            if (symbol == 1) {
                N = N - 1;
            }
        } else {
            auto w = idx.m_cst.select_child(node, 1);
            int root_id = idx.m_cst.id(idx.m_cst.root());
            while (idx.m_cst.id(w) != root_id) {
                int symbol = idx.m_cst.edge(w, pat_size + 1);
                if (symbol != 1) {
                    uint64_t leftbound = idx.m_cst.lb(w);
                    uint64_t rightbound = idx.m_cst.rb(w);
                    freq = rightbound - leftbound + 1;
                    if (freq == 1 && rightbound != leftbound) {
                        freq = 0;
                    }
                    if (freq == 1)
                        N1 += 1;
                    else if (freq == 2)
                        N2 += 1;
                    else if (freq >= 3)
                        N3 += 1;
                }
                w = idx.m_cst.sibling(w);
            }
        }
    } else {
        int symbol = idx.m_cst.edge(node, pat.size() + 1);
        if (!ismkn) {
            if (symbol != 1) {
                N = 1;
            }
        }
        if (ismkn) {
            if (symbol != 1) {
                if (ismkn) {
                    uint64_t leftbound = idx.m_cst.lb(node);
                    uint64_t rightbound = idx.m_cst.rb(node);
                    freq = rightbound - leftbound + 1;

                    if (freq == 1 && rightbound != leftbound) {
                        freq = 0;
                    }
                    if (freq == 1)
                        N1 += 1;
                    else if (freq == 2)
                        N2 += 1;
                    else if (freq >= 3)
                        N3 += 1;
                }
            }
        }
    }
}

template <class t_idx>
int N1PlusBack(const t_idx& idx, uint64_t lb, uint64_t rb, std::vector<uint64_t> patrev)
{
    int c = 0;
    auto node = idx.m_cst_rev.node(lb, rb);
    int patrev_size = patrev.size();
    int deg = idx.m_cst_rev.degree(node);
    if (patrev_size == idx.m_cst_rev.depth(node)) {
        c = deg;
        auto w = idx.m_cst_rev.select_child(node, 1);
        int symbol = idx.m_cst_rev.edge(w, patrev_size + 1);
        if (symbol == 1)
            c = c - 1;
    } else {
        int symbol = idx.m_cst_rev.edge(node, patrev_size + 1);
        if (symbol != 1) {
            c = 1;
        }
    }
    return c;
}

template <class t_idx>
double pkn(const t_idx& idx, std::vector<uint64_t> pat)
{
    int size = pat.size();
    double probability = 0;

    if ((size == ngramsize && ngramsize != 1) || (pat[0] == STARTTAG)) { //for the highest order ngram, or the ngram that starts with <s>
        std::vector<uint64_t> pat2 = pat;
        pat2.erase(pat2.begin());
        double backoff_prob = pkn(idx, pat2);
        denominator = 0;
        int c = 0;
        uint64_t lb = 0, rb = idx.m_cst.size() - 1;
        backward_search(idx.m_cst_rev.csa, lb, rb, pat.rbegin(), pat.rend(), dot_LB, dot_RB);
        c = rb - lb + 1;
        if (c == 1 && lb != rb) {
            c = 0;
        }
        double D = discount(idx, c);

        double numerator = 0;
        if (c - D > 0) {
            numerator = c - D;
        }

        pat.pop_back();
        lb = 0;
        rb = idx.m_cst.size() - 1;
        backward_search(idx.m_cst.csa, lb, rb, pat.begin(), pat.end(), dot_LB_dot, dot_RB_dot);
        freq = rb - lb + 1;
        if (freq == 1 && lb != rb) {
            freq = 0;
        }
        denominator = freq;
        if (denominator == 0) {
            cout << "---- Undefined fractional number XXXZ - Backing-off ---" << endl;
            double output = backoff_prob; //TODO check this
            return output;
        }

        if (freq > 0) {
            N1PlusFront(idx, lb, rb, pat);
        }
        if (ismkn) {
            double gamma = (idx.m_D1[ngramsize] * N1) + (idx.m_D2[ngramsize] * N2) + (idx.m_D3[ngramsize] * N3);
            double output = (numerator / denominator) + (gamma / denominator) * backoff_prob;
            return output;
        } else {
            double output = (numerator / denominator) + (D * N / denominator) * backoff_prob;
            return output;
        }
    } else if (size < ngramsize && size != 1) { //for lower order ngrams

        std::vector<uint64_t> pat2 = pat;
        pat2.erase(pat2.begin());
        double backoff_prob = pkn(idx, pat2);
        denominator = 0;
        int c = 0;
        uint64_t lbrev = 0, rbrev = idx.m_cst_rev.size() - 1;
        backward_search(idx.m_cst_rev.csa, lbrev, rbrev, pat.rbegin(), pat.rend(), dot_LB, dot_RB);
        dot_LB = lbrev;
        dot_RB = rbrev;
        freq = rbrev - lbrev + 1;
        if (freq == 1 && lbrev != rbrev) {
            freq = 0;
        }

        std::vector<uint64_t> patrev = pat; //TODO replace this
        reverse(patrev.begin(), patrev.end());
        if (freq > 0) {
            c = N1PlusBack(idx, lbrev, rbrev, patrev); //TODO fix this
        }
        double D = discount(idx, freq);

        double numerator = 0;
        if (c - D > 0) {
            numerator = c - D;
        }

        pat.pop_back();

        uint64_t lb = 0, rb = idx.m_cst.size() - 1;
        //takes care of the first call - Bigram level
        if (dot_LB_dot == -5 && dot_RB_dot == -5) {
            dot_LB_dot = lb;
            dot_RB_dot = rb;
        }
        backward_search(idx.m_cst.csa, lb, rb, pat.begin(), pat.end(), dot_LB_dot, dot_RB_dot);
        dot_LB_dot = lb;
        dot_RB_dot = rb;
        freq = rb - lb + 1;
        if (freq == 1 && lb != rb) {
            freq = 0;
        }
        double denominator = 0;
        if (freq != 1) {
            N1PlusFrontBack_Front(idx, pat, lb, rb);
        } else {
            denominator = 1;
            N = 1; //TODO fix this
        }
        if (denominator == 0) {
            cout << "---- Undefined fractional number XXXW-backing-off---" << endl;
            double output = backoff_prob;
            return output;
        }

        if (ismkn) {
            double gamma = 0;
            if (freq > 0) {
                N1PlusFront(idx, lb, rb, pat);
            }
            gamma = (idx.m_D1[size] * N1) + (idx.m_D2[size] * N2) + (idx.m_D3[size] * N3);
            double output = numerator / denominator + (gamma / denominator) * backoff_prob;
            return output;
        } else {
            double output = (numerator / denominator) + (D * N / denominator) * backoff_prob;
            return output;
        }
    } else if (size == 1 || ngramsize == 1) //for unigram
    {
        denominator = 0;
        uint64_t lbrev = 0, rbrev = idx.m_cst_rev.size() - 1;
        backward_search(idx.m_cst_rev.csa, lbrev, rbrev, pat.begin(), pat.end(), lbrev, rbrev);
        dot_LB = lbrev;
        dot_RB = rbrev;
        denominator = idx.m_N1plus_dotdot;
        int c = N1PlusBack(idx, lbrev, rbrev, pat);
        if (!ismkn) {
            double output = c / denominator;
            cout << "Lowest Order "
                 << " numerator is: " << c << " denomiator is: " << denominator << endl;
            cout << "Lowest Order probability " << output << endl;

            return output;
        } else {

            double D = discount(idx, freq);

            double numerator = 0;
            if (c - D > 0) {
                numerator = c - D;
            }

            double gamma = 0;
            N1 = idx.m_n1[1];
            N2 = idx.m_n2[1];
            N3 = idx.m_N3plus_dot;
            gamma = (idx.m_D1[size] * N1) + (idx.m_D2[size] * N2) + (idx.m_D3[size] * N3);
            double output = numerator / denominator + (gamma / denominator) * (1 / (double)idx.vocab_size());
            return output;
        }
    }
    return probability;
}

template <class t_idx>
double run_query_knm(const t_idx& idx, const std::vector<uint64_t>& word_vec)
{
    double final_score = 0;
    std::deque<uint64_t> pattern_deq;
    for (const auto& word : word_vec) {
        pattern_deq.push_back(word);
        if (word == STARTTAG)
            continue;
        if (pattern_deq.size() > ngramsize) {
            pattern_deq.pop_front();
        }
        std::vector<uint64_t> pattern(pattern_deq.begin(), pattern_deq.end());
        double score = pkn(idx, pattern);
        final_score += log10(score);
    }
    return final_score;
}

template <class t_idx>
void run_queries(const t_idx& idx, const std::vector<std::vector<uint64_t> > patterns)
{
    using clock = std::chrono::high_resolution_clock;
    double perplexity = 0;
    int M = 0;
    std::chrono::nanoseconds total_time(0);
    for (std::vector<uint64_t> pattern : patterns) {

        pattern.push_back(ENDTAG);
        pattern.insert(pattern.begin(), STARTTAG);
        M += pattern.size() + 1; // +1 for adding </s>
        // run the query
        auto start = clock::now();
        double sentenceprob = run_query_knm(idx, pattern);
        auto stop = clock::now();
        perplexity += log10(sentenceprob);
        // output score
        std::copy(pattern.begin(), pattern.end(), std::ostream_iterator<uint64_t>(std::cout, " "));
        std::cout << " -> " << sentenceprob << endl;
        total_time += (stop - start);
    }
    std::cout << "time in milliseconds = "
              << std::chrono::duration_cast<std::chrono::microseconds>(total_time).count() / 1000.0f
              << " ms" << endl;
    perplexity = (1 / (double)M) * perplexity;
    perplexity = pow(10, (-perplexity));
    std::cout << "Perplexity = " << perplexity << endl;
}

int main(int argc, const char* argv[])
{
    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    ngramsize = args.ngramsize;
    ismkn = args.ismkn;

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    /* load index */
    using csa_type = sdsl::csa_sada_int<>;
    using cst_type = sdsl::cst_sct3<csa_type>;
    index_succinct<cst_type> idx;

    auto index_file = args.collection_dir + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if (utils::file_exists(index_file)) {
        std::cout << "loading index from file '" << index_file << "'" << std::endl;
        sdsl::load_from_file(idx, index_file);
    } else {
        std::cerr << "index does not exist. build it first" << std::endl;
        return EXIT_FAILURE;
    }

    /* print precomputed parameters */

    for (int size = 1; size <= ngramsize; size++) {
        cout << idx.m_n1[size] << " "; ////XXXXX fails
    }
    cout << endl;
    cout << "------------------------------------------------" << endl;
    cout << idx.m_N1plus_dotdot << endl;
    cout << idx.m_N3plus_dot << endl;
    cout << "------------------------------------------------" << endl;

    /* parse pattern file */
    std::vector<std::vector<uint64_t> > patterns;
    if (utils::file_exists(args.pattern_file)) {
        std::ifstream ifile(args.pattern_file);
        std::cout << "reading input file '" << args.pattern_file << "'" << std::endl;
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
        std::cerr << "cannot read pattern file '" << args.pattern_file << "'" << std::endl;
        return EXIT_FAILURE;
    }

    {
        //        run_queries(idx, patterns);
    }
    return 0;
}
