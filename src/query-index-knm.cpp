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
const int STARTTAG = 3;
const int ENDTAG = 4;

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

void print(const std::vector<uint64_t>::iterator& pattern_begin,
           const std::vector<uint64_t>::iterator& pattern_end)
{
        for(auto it = pattern_begin;it!=pattern_end;it++)
        {
                cout<<*it<<" ";
        }
        cout<<endl;
}

// computes N_1+( * abc ) equivalent to computing N_1+ ( cba *) in the reverse suffix tree
template <class t_idx>
int N1PlusBack(const t_idx& idx, const uint64_t& lb_rev, const uint64_t& rb_rev, int patrev_size, bool check_for_EOS=true)
{
    uint64_t c = 0;
    auto node = idx.m_cst_rev.node(lb_rev, rb_rev);
    uint64_t deg = idx.m_cst_rev.degree(node);
    if (patrev_size == idx.m_cst_rev.depth(node)) {
        c = deg;
        if (check_for_EOS) {
        	auto w = idx.m_cst_rev.select_child(node, 1);
        	uint64_t symbol = idx.m_cst_rev.edge(w, patrev_size + 1);
        	if (symbol == 1)
            		c = c - 1;
	}
    } else {
	if (check_for_EOS) {
        	uint64_t symbol = idx.m_cst_rev.edge(node, patrev_size + 1); 
        	if (symbol != 1) 
        	    c = 1;
	} else {
		c = 1;
	}
    }
    return c;
}

template <class t_idx>
double discount(const t_idx& idx)
{
    double D = idx.m_Y[ngramsize];
    return D;
}

//  Computes N_1+( * ab * ) 
//  n1plus_front = value of N1+( * abc ) (for some following symbol 'c')
//  if this is N_1+( * ab ) = 1 then we know the only following symbol is 'c'
//  and thus N1+( * ab * ) is the same as N1+( * abc ), stored in n1plus_front
template <class t_idx>
uint64_t N1PlusFrontBack(const t_idx& idx, 
                     const uint64_t& lb, const uint64_t& rb, 
                     const uint64_t n1plus_front,
                     std::vector<uint64_t>::reverse_iterator pattern_rbegin,
                     std::vector<uint64_t>::reverse_iterator pattern_rend,
		     bool check_for_EOS=true)
{
    // ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
    // ASSUMPTION: pattern_begin, pattern_end cover just the pattern we're interested in (i.e., we want N1+ dot pattern dot)
    int pattern_size = std::distance(pattern_rbegin,pattern_rend);
    auto node = idx.m_cst.node(lb, rb);
    uint64_t back_N1plus_front = 0;

    // this is when the pattern matches a full edge in the CST
    if (pattern_size == idx.m_cst.depth(node)) {
        auto w = idx.m_cst.select_child(node, 1);

        int root_id = idx.m_cst.id(idx.m_cst.root());
    	while (idx.m_cst.id(w) != root_id) {
        	uint64_t symbol = idx.m_cst.edge(w, pattern_size + 1);
        	if (symbol != 1 || !check_for_EOS) {
		    uint64_t lb_rev_prime = 0, rb_rev_prime = idx.m_cst_rev.size()-1;//full interval - very inefficient
		    // first step: find the symbol to the right
		    // (which is first in the reverse order)
		   backward_search(idx.m_cst_rev.csa, 
				   lb_rev_prime, rb_rev_prime, 
				   symbol,
				   lb_rev_prime, rb_rev_prime);  
		// this is a full search for the pattern!
		   backward_search(idx.m_cst_rev.csa, 
		                   lb_rev_prime, rb_rev_prime, 
				   pattern_rbegin, 
				   pattern_rend,
				   lb_rev_prime, rb_rev_prime);  
        	    back_N1plus_front += N1PlusBack(idx, lb_rev_prime, rb_rev_prime, pattern_size+1, check_for_EOS);
       		}
       		w = idx.m_cst.sibling(w);
    	}         
    	return back_N1plus_front;
    } else {
        // special case, only one way of extending this pattern to the right
	return n1plus_front;
    }
}

// Computes N_1+( abc * ) 
template <class t_idx>
uint64_t N1PlusFront(const t_idx& idx, 
                     const uint64_t& lb, const uint64_t& rb, 
                     std::vector<uint64_t>::iterator pattern_begin,
                     std::vector<uint64_t>::iterator pattern_end,
	             bool check_for_EOS=true)
{
    // ASSUMPTION: lb, rb already identify the suffix array range corresponding to 'pattern' in the forward tree
    auto node = idx.m_cst.node(lb, rb);
    int pattern_size = std::distance(pattern_begin,pattern_end);
    uint64_t N1plus_front = 0;
    if (pattern_size == idx.m_cst.depth(node)) {
        auto w = idx.m_cst.select_child(node, 1);
        N1plus_front = idx.m_cst.degree(node);
        if (check_for_EOS) {
	    uint64_t symbol = idx.m_cst.edge(w, pattern_size + 1);
	    if (symbol == 1) {
		N1plus_front = N1plus_front - 1;
	    }
	}
        return N1plus_front;
    } else {
        if (check_for_EOS) {
	    uint64_t symbol = idx.m_cst.edge(node, pattern_size + 1);
	    if (symbol != 1) {
		N1plus_front = 1;
	    }
	}
	return N1plus_front;
    }
}

template <class t_idx>
double highestorder(const t_idx& idx, 
		    const std::vector<uint64_t>::iterator& pattern_begin,
                    const std::vector<uint64_t>::iterator& pattern_end,
                    const std::vector<uint64_t>::reverse_iterator& pattern_rbegin,
                    const std::vector<uint64_t>::reverse_iterator& pattern_rend,
 	            uint64_t& lb, uint64_t& rb,
                    uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
    double backoff_prob = pkn(idx, pattern_begin+1, pattern_end,
                              pattern_rbegin, pattern_rend-1,
 			      lb, rb,
			      lb_rev, rb_rev, char_pos,d);
    auto node = idx.m_cst_rev.node(lb_rev,rb_rev);
    uint64_t denominator = 0;
    uint64_t c = 0;
    cout<<"SYMBOL="<<*pattern_begin<<" d="<<d<<endl;
    if(forward_search(idx.m_cst_rev, node, d, *pattern_begin, char_pos)>0){
	lb_rev = idx.m_cst_rev.lb(node);
    	rb_rev = idx.m_cst_rev.rb(node);
    	c = rb_rev - lb_rev + 1;
    }

    double D = discount(idx);

    double numerator = 0;
    if (c - D > 0) {
        numerator = c - D;
    }

    uint64_t N1plus_front = 0;
    if(backward_search(idx.m_cst.csa, lb, rb, *pattern_begin, lb, rb)>0){
	denominator = rb - lb + 1;
        N1plus_front = N1PlusFront(idx, lb, rb, pattern_begin, pattern_end - 1);
    }else{
        cout << "---- Undefined fractional number XXXZ - Backing-off ---" << endl;
        return backoff_prob; 
    }

    double output = (numerator / denominator) + (D * N1plus_front / denominator) * backoff_prob;
    cout
    << " Highest Order" << endl
    << " N1plus_front is: " << N1plus_front << endl
    << " D is: " << D << endl
    << " numerator is: " << numerator << endl 
    << " denomiator is: " << denominator << endl
    << " Highest Order probability " << output << endl
    << "------------------------------------------------" << endl;
    return output;
}

template <class t_idx>
double lowerorder(const t_idx& idx,
                  const std::vector<uint64_t>::iterator& pattern_begin,
                  const std::vector<uint64_t>::iterator& pattern_end,
                  const std::vector<uint64_t>::reverse_iterator& pattern_rbegin,
                  const std::vector<uint64_t>::reverse_iterator& pattern_rend,
                  uint64_t& lb, uint64_t& rb,
                  uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
    double backoff_prob = pkn(idx, pattern_begin+1, pattern_end,
                              pattern_rbegin, pattern_rend-1,
                              lb, rb,
                              lb_rev, rb_rev, char_pos,d);
    
    uint64_t c = 0;
    auto node = idx.m_cst_rev.node(lb_rev,rb_rev);
    int pattern_size = std::distance(pattern_begin, pattern_end);
    print(pattern_begin,pattern_end);
    if(forward_search(idx.m_cst_rev, node, d , *(pattern_begin), char_pos)>0){
        lb_rev = idx.m_cst_rev.lb(node);
        rb_rev = idx.m_cst_rev.rb(node);
        c = N1PlusBack(idx, lb_rev, rb_rev, pattern_size);
    }

    double D = discount(idx);
    double numerator = 0;
    if (c - D > 0) {
        numerator = c - D;
    }

    uint64_t N1plus_front = 0;
    uint64_t back_N1plus_front = 0;
    if(backward_search(idx.m_cst.csa, lb, rb,*(pattern_begin) , lb, rb)>0){//TODO CHECK: what happens to the bounds when this is false?
        back_N1plus_front = N1PlusFrontBack(idx, lb, rb, c, pattern_rbegin, pattern_rend);
	N1plus_front = N1PlusFront(idx, lb, rb, pattern_begin, pattern_end-1);
    }else{
        cout << "---- Undefined fractional number XXXZ - Backing-off ---" << endl;
        return backoff_prob;
    }

    d++;
    double output = (numerator / back_N1plus_front) + (D * N1plus_front / back_N1plus_front) * backoff_prob;
    cout 
    << "Lower Order" << endl
    << " N1plus_front is: " << N1plus_front << endl
    << " D is: " << D << endl
    << " numerator is: " << numerator << endl
    << " back_N1plus_front is: " << back_N1plus_front << endl
    << " Lower Order probability " << output << endl
    << "------------------------------------------------" << endl;
    return output;
}

template <class t_idx>
double lowestorder(const t_idx& idx, 
		   const uint64_t& pattern,
                   uint64_t& lb_rev, uint64_t& rb_rev, uint64_t& char_pos, uint64_t& d)
{
    auto node = idx.m_cst_rev.node(lb_rev,rb_rev);
    double denominator = 0;
    cout<<"SYMBOL="<<pattern<<" d="<<d<<endl;
    forward_search(idx.m_cst_rev, node, d, pattern, char_pos);
    d++;
    denominator = idx.m_N1plus_dotdot;
    lb_rev = idx.m_cst_rev.lb(node);
    rb_rev = idx.m_cst_rev.rb(node);
    int numerator = N1PlusBack(idx, lb_rev, rb_rev, 1);//TODO precompute this
    double probability = (double)numerator / denominator;

    cout 
    << "Lowest Order numerator is: "<< numerator << endl 
    << " denomiator is: " << denominator << endl 
    << " Lowest Order probability " << probability << endl 
    << "------------------------------------------------" << endl;

    return probability;
}

template <class t_idx>
double pkn(const t_idx& idx, 
	   const std::vector<uint64_t>::iterator& pattern_begin, 
           const std::vector<uint64_t>::iterator& pattern_end,
           const std::vector<uint64_t>::reverse_iterator& pattern_rbegin,
	   const std::vector<uint64_t>::reverse_iterator& pattern_rend,
	   uint64_t& lb, uint64_t& rb,
           uint64_t &lb_rev,uint64_t &rb_rev, uint64_t& char_pos, uint64_t& d)
{
    int size = std::distance(pattern_begin,pattern_end);
    double probability = 0;
    if ((size == ngramsize && ngramsize != 1) || (*pattern_begin == STARTTAG)) {
	print(pattern_begin,pattern_end);
        cout<<".("<<lb_rev<<","<<rb_rev<<")."<<endl;
        probability = highestorder(idx, pattern_begin, pattern_end, 
                                   pattern_rbegin, pattern_rend,
				   lb, rb, 
				   lb_rev, rb_rev, char_pos, d);
        cout<<".("<<lb_rev<<","<<rb_rev<<")."<<endl;
    } else if (size < ngramsize && size != 1) {
        print(pattern_begin,pattern_end);
        if(size==0)exit(1);
        cout<<"..("<<lb_rev<<","<<rb_rev<<").."<<endl;

        probability = lowerorder(idx, pattern_begin, pattern_end,
                                 pattern_rbegin, pattern_rend,
				 lb, rb,
				 lb_rev, rb_rev, char_pos, d);

        cout<<"..("<<lb_rev<<","<<rb_rev<<").."<<endl;
    } else if (size == 1 || ngramsize == 1) {
        cout<<"...("<<lb_rev<<","<<rb_rev<<")..."<<endl;
        probability = lowestorder(idx, *(pattern_end-1), 
			  	  lb_rev, rb_rev, char_pos, d);
        cout<<"...("<<lb_rev<<","<<rb_rev<<")..."<<endl;
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

        uint64_t lb_rev = 0, rb_rev = idx.m_cst_rev.size() - 1, lb=0, rb=idx.m_cst.size() - 1;
        uint64_t char_pos=0, d=0;
        double score = pkn(idx, pattern.begin(),pattern.end(), 
			   pattern.rend(), pattern.rbegin(),
			   lb, rb, 
			   lb_rev, rb_rev, char_pos, d);
        final_score += log10(score);
        cout << "-----------------------------------------------" << endl;
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
        M += pattern.size() + 1; // +1 for adding </s>
        pattern.push_back(ENDTAG);
        pattern.insert(pattern.begin(), STARTTAG);
        // run the query
        auto start = clock::now();
        double sentenceprob = run_query_knm(idx, pattern);
        auto stop = clock::now();
        perplexity += sentenceprob;
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

