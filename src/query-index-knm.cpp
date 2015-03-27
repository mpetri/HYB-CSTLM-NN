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

int ngramsize;
bool ismkn;
uint64_t STARTTAG = 3;
uint64_t ENDTAG = 4;

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


// computes N_1+( * abc )
template <class t_idx>
int N1PlusBack(const t_idx& idx, uint64_t lb, uint64_t rb, int patrev_size)
{
    int c = 0;
    auto node = idx.m_cst_rev.node(lb, rb);
    int deg = idx.m_cst_rev.degree(node);
    if (patrev_size == idx.m_cst_rev.depth(node)) {
        c = deg;
        auto w = idx.m_cst_rev.select_child(node, 1);
        int symbol = idx.m_cst_rev.edge(w, patrev_size + 1);
        if (symbol == 1)
            c = c - 1;
    } else {
        int symbol = idx.m_cst_rev.edge(node, patrev_size + 1); // trevor is suspicious about the symmetry with the above test!
        if (symbol != 1) 
            c = 1;
    }
    return c;
}

template <class t_idx>
int discount(const t_idx& idx, int c)
{
    double D = idx.m_Y[ngramsize];
    return D;
}

template <class t_idx>
double highestorder(const t_idx& idx, 
		    const std::vector<uint64_t>::iterator& pattern_begin,
                    const std::vector<uint64_t>::iterator& pattern_end,
 	            uint64_t& lb_rev_prime, uint64_t& rb_rev_prime, uint64_t& char_pos_prime, uint64_t& d_prime,
                    uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
    int size = std::distance(pattern_begin,pattern_end);
    double backoff_prob = pkn(idx, (pattern_begin+1), pattern_end, 
 			      lb_rev_prime, rb_rev_prime, char_pos_prime, d_prime,
			      lb_rev, rb_rev, char_pos,d);
    auto node = idx.m_cst_rev.node(lb_rev,rb_rev);
    uint64_t denominator = 0;
    uint64_t c = 0;
    cout<<"SYMBOL="<<*pattern_begin<<" d="<<d<<endl;
    forward_search(idx.m_cst_rev, node, d , *pattern_begin, char_pos);//FIXME d should be fixed
    lb_rev = idx.m_cst_rev.lb(node);
    rb_rev = idx.m_cst_rev.rb(node);
    c = rb_rev - lb_rev + 1;
    if (c == 1 && lb_rev != rb_rev) {
        c = 0;
    }

    double D = discount(idx, c);

    double numerator = 0;
    if (c - D > 0) {
        numerator = c - D;
    }
    return 0;
}

template <class t_idx>
double lowestorder(const t_idx& idx, 
		   const std::vector<uint64_t>::iterator& pattern_begin, 
                   const std::vector<uint64_t>::iterator& pattern_end,
		   uint64_t& lb_rev_prime, uint64_t& rb_rev_prime, uint64_t& char_pos_prime, uint64_t& d_prime,
                   uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
    auto node = idx.m_cst_rev.node(lb_rev,rb_rev);
    double denominator = 0;
    int pattern_size = std::distance(pattern_begin, pattern_end);
    cout<<"SYMBOL="<<*pattern_begin<<" d="<<d<<endl;
    forward_search(idx.m_cst_rev, node, d, *(pattern_end-1), char_pos);
    d++;
    denominator = idx.m_N1plus_dotdot;
    lb_rev = idx.m_cst_rev.lb(node);
    rb_rev = idx.m_cst_rev.rb(node);
    int c = N1PlusBack(idx, lb_rev, rb_rev, pattern_size);//TODO precompute this
    double probability = (double)c / denominator;
/*
    cout << "Lowest Order numerator is: "
    << c << " denomiator is: " << denominator << endl;
    cout << "Lowest Order probability " << probability << endl;
    cout << "------------------------------------------------" << endl;
*/
    return probability;
}

void print(std::vector<uint64_t>::iterator pattern_begin,
           std::vector<uint64_t>::iterator pattern_end)
{
	for(auto it = pattern_begin;it!=pattern_end;it++)
        {
		cout<<*it<<" ";
	}
	cout<<endl;
}

template <class t_idx>
double pkn(const t_idx& idx, 
	   const std::vector<uint64_t>::iterator& pattern_begin, 
           const std::vector<uint64_t>::iterator& pattern_end,
	   uint64_t& lb_rev_prime, uint64_t& rb_rev_prime, uint64_t& char_pos_prime, uint64_t& d_prime,
           uint64_t &lb_rev,uint64_t &rb_rev, uint64_t& char_pos, uint64_t& d)
{
    int size = std::distance(pattern_begin,pattern_end);
    double probability = 0;
    if ((size == ngramsize && ngramsize != 1) || (*pattern_begin == STARTTAG)) {
        cout<<".("<<lb_rev<<","<<rb_rev<<")."<<endl;
        probability = highestorder(idx, pattern_begin, pattern_end, 
				   lb_rev_prime, rb_rev_prime, char_pos_prime, d_prime, 
				   lb_rev, rb_rev, char_pos, d);
        cout<<"HIGHEST"<<endl;
        print(pattern_begin,pattern_end);
        cout<<"..("<<lb_rev<<","<<rb_rev<<").."<<endl;
    } else if (size < ngramsize && size != 1) {
       // probability = lowerorder(idx, pat, size);
    } else if (size == 1 || ngramsize == 1) {
        cout<<"...("<<lb_rev<<","<<rb_rev<<")..."<<endl;
        probability = lowestorder(idx, pattern_begin, pattern_end, 
				  lb_rev_prime, rb_rev_prime, char_pos_prime, d_prime, 
			  	  lb_rev, rb_rev, char_pos, d);
        cout<<"LOWEST"<<endl;
        print(pattern_begin,pattern_end);
        cout<<"("<<lb_rev<<","<<rb_rev<<")"<<endl;
    }
    return probability;
}

template <class t_idx>
double run_query_knm(const t_idx& idx, const std::vector<uint64_t>& word_vec)
{
    double final_score = 0;
    std::deque<uint64_t> pattern_deq;
    for (const auto& word : word_vec) {
        cout << "------------------------------------------------" << endl;
        pattern_deq.push_back(word);
        if (word == STARTTAG)
            continue;
        if (pattern_deq.size() > ngramsize) {
            pattern_deq.pop_front();
        }
        std::vector<uint64_t> pattern(pattern_deq.begin(), pattern_deq.end());
        cout << "PATTERN is = ";
        for (int i = 0; i < pattern.size(); i++)
            cout << pattern[i] << " ";
        cout << endl;

        uint64_t lb_rev = 0, rb_rev = idx.m_cst_rev.size() - 1, lb_rev_prime=0, rb_rev_prime=idx.m_cst_rev.size() - 1;
        uint64_t char_pos=0, d=0, char_pos_prime=0, d_prime=0;
        double score = pkn(idx, pattern.begin(),pattern.end(),
			   lb_rev_prime, rb_rev_prime, char_pos_prime, d_prime, 
			   lb_rev, rb_rev, char_pos, d);
        final_score += log10(score);
        cout << "------------------------------------------------" << endl;
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
    cout << "------------------------------------------------" << endl;
    cout << "-------------PRECOMPUTED QUANTITIES-------------" << endl;
    cout << "------------------------------------------------" << endl;
    cout << "n1 = ";
    for (int size = 0; size <= ngramsize; size++) {
        cout << idx.m_n1[size] << " ";
    }
    cout << endl;
    cout << "n2 = ";
    for (int size = 0; size <= ngramsize; size++) {
        cout << idx.m_n2[size] << " ";
    }
    cout << endl;
    cout << "n3 = ";
    for (int size = 0; size <= ngramsize; size++) {
        cout << idx.m_n3[size] << " ";
    }
    cout << endl;
    cout << "n4 = ";
    for (int size = 0; size <= ngramsize; size++) {
        cout << idx.m_n4[size] << " ";
    }
    cout << endl;
    cout << "------------------------------------------------" << endl;
    cout << "Y = ";
    for (int size = 0; size <= ngramsize; size++) {
        cout << idx.m_Y[size] << " ";
    }
    if (ismkn) {
        cout << endl;
        cout << "D1 = ";
        for (int size = 0; size <= ngramsize; size++) {
            cout << idx.m_D1[size] << " ";
        }
        cout << endl;
        cout << "D2 = ";
        for (int size = 0; size <= ngramsize; size++) {
            cout << idx.m_D2[size] << " ";
        }
        cout << endl;
        cout << "D3+= ";
        for (int size = 0; size <= ngramsize; size++) {
            cout << idx.m_D3[size] << " ";
        }
    }
    cout << endl;
    cout << "------------------------------------------------" << endl;
    cout << "N1+(..) = " << idx.m_N1plus_dotdot << endl;
    cout << "N3+(.) = " << idx.m_N3plus_dot << endl;//FIXME
    cout << "------------------------------------------------" << endl;
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
        run_queries(idx, patterns);
    }
    return 0;
}
