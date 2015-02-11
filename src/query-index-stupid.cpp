#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"
#include <sdsl/suffix_array_algorithm.hpp>
#include <iostream>

#include "utils.hpp"
#include "collection.hpp"
#include "index_succinct.hpp"

typedef struct cmdargs {
    std::string pattern_file;
    std::string collection_dir;
} cmdargs_t;

void
print_usage(const char* program)
{
    fprintf(stdout,"%s -c <collection dir> -p <pattern file>\n",program);
    fprintf(stdout,"where\n");
    fprintf(stdout,"  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout,"  -p <pattern file>  : the pattern file.\n");
};

cmdargs_t
parse_args(int argc,const char* argv[])
{
    cmdargs_t args;
    int op;
    args.pattern_file = "";
    args.collection_dir = "";
    while ((op=getopt(argc,(char* const*)argv,"p:c:")) != -1) {
        switch (op) {
            case 'p':
                args.pattern_file = optarg;
                break;
            case 'c':
                args.collection_dir = optarg;
                break;
        }
    }
    if (args.collection_dir==""||args.pattern_file=="") {
        std::cerr << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

double d = 0.4;
double stupidbackoff(const csa_wt<wt_huff_int<>> &csarev, const deque<int> &pattern_rev)
{
    int N = csarev.size()-1;    //size of the suffix array
    int M = pattern_rev.size(); //size of the pattern
    // assumes that the pattern is reversed
    
    // create a copy of the pattern with and without first (=last) entry
    vector<int> full_pattern, context_pattern;
    full_pattern.push_back(pattern_rev[0]);
    
    // track the lower and upper bounds for numerator (full) and denominator
    // (context) matches
    uint64_t lb_num=0,lb_denom=0, rb_num=N-1,rb_denom=N-1;
    uint64_t lb_num_prev=0,lb_denom_prev=0, rb_num_prev=N-1,rb_denom_prev=N-1;
    double s=0;
    for (int m=1; m<=M; m++)
    {
        double numer=0, denom=0;

	lb_num_prev = lb_num;
	rb_num_prev = rb_num;
        backward_search(csarev, lb_num, rb_num,
                        full_pattern.begin(), full_pattern.end(),
                        lb_num, rb_num);
        
	numer = rb_num - lb_num + 1;

	// missing patterns || unknown words
        if(lb_num>rb_num || (lb_num<0 || rb_num>N))
        {
            s *= pow(d,min(M-m, 5));
            // TODO use a smarter backoff weighting
            break;
        }
	
	rb_denom_prev = rb_denom;
	lb_denom_prev = lb_denom;
        
        if(m>=2)
        {
                backward_search(csarev, lb_denom, rb_denom,
                           context_pattern.begin(), context_pattern.end(),
                            lb_denom, rb_denom);
        }
	
        denom = rb_denom - lb_denom + 1;
        s = numer/denom;
	
	if(lb_num!=rb_num)
        {
		full_pattern[0]=pattern_rev[m];
	}else
	{
		//re-use the previous search interval
		lb_num = lb_num_prev;
		rb_num = rb_num_prev;
		//grow the pattern
		full_pattern.push_back(pattern_rev[m]);
	}

	if(lb_denom!=rb_denom)
	{
		if(context_pattern.size()!=0)
			context_pattern[0]=pattern_rev[m];
		else
			context_pattern.push_back(pattern_rev[m]);
	}else
	{
		//re-use the previous search interval
		lb_denom = lb_denom_prev;
		rb_denom = rb_denom_prev;
		//grow the pattern
        	context_pattern.push_back(pattern_rev[m]);
	}
    }
    return s;
}

template<class t_idx>
std::vector<std::pair<uint64_t,uint64_t>>
run_query_stupid(const t_idx& idx,const std::vector<uint64_t>& tokens)
{
    std::vector<std::pair<uint64_t,uint64_t>> res;
	size_t lb=0;
	size_t rb=idx.m_cst.size()-1;
	size_t res_lb = 0;
	size_t res_rb = 0;
	auto itr = tokens.rbegin();
	auto end = tokens.rend();
	size_t len = 0;
	while(itr != end) {
		typename t_idx::csa_type::char_type chr = *itr;
		sdsl::backward_search(idx.m_cst.csa,lb,rb,chr,res_lb,res_rb);
		if(res_lb > res_rb) break;
        res.emplace_back(len+1,res_rb-res_lb+1);
		lb = res_lb; rb = res_rb;
		itr++;
		len++;
	}
    return res;
}


int main(int argc,const char* argv[])
{
	using clock = std::chrono::high_resolution_clock;

    /* parse command line */
    cmdargs_t args = parse_args(argc,argv);

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    /* load index */
    using csa_type = sdsl::csa_sada_int<>;
    using cst_type = sdsl::cst_sct3<csa_type>;
    index_succinct<cst_type> idx;
    auto index_file = args.collection_dir + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    std::cout << "loading index from file '" << index_file << "'" << std::endl;
    sdsl::load_from_file(idx,index_file);


    /*
    while (getline(file,line))
    {
        vector<int> word_vec;
        istringstream iss(line);
        string word;
        int size=1;
        double score=1;
        
        //loads the a test sentence into a vector of words
        while (std::getline(iss, word, ' '))
        {
            int num = stoi(word);
            word_vec.push_back(num);
        }
        
        //generates all the required patterns from the reverse word vector: c b a -> c, c b , c b a
        deque<int> pattern;
        for (auto it = word_vec.begin(); it != word_vec.end(); ++it)
        {
            pattern.push_front(*it);
            while (ngramsize > 0 && pattern.size() > ngramsize)
                pattern.pop_back();

            //calls stupidbackoff to get the score for the pattern
            //Example: score (c b a) = stupidbackoff(c)*stupidbackoff(c b)*stupidbackoff(c b a)
            double dummyscore = stupidbackoff(csarev, pattern);
	    score*=dummyscore;
	//    cout<<"SCORE: "<<dummyscore<<endl;;
            //TODO reuse numerator of the previous pattern
        }
    cout<<"FINAL SCORE: "<<score<<endl;
    }		
    */

    /* parse pattern file */
    std::chrono::nanoseconds total_time(0);
    if(utils::file_exists(args.pattern_file)) {
    	std::ifstream ifile(args.pattern_file);
    	std::cout << "reading input file '" << args.pattern_file << "'" << std::endl;
    	std::string line;
    	while (std::getline(ifile,line)) {
    		std::vector<uint64_t> tokens;
    		std::istringstream iss(line);
    		std::string word;
    		while (std::getline(iss, word, ' ')) {
    			uint64_t num = std::stoull(word);
    			tokens.push_back(num);
    		}

    		// run the query
    		auto start = clock::now();
    		auto res = run_query_stupid(idx,tokens);
    		auto stop = clock::now();
            for(const auto& r : res) {
                std::cout << "len = " << r.first << " freq = " <<  r.second << std::endl;
            }
    		total_time += (stop-start);
    	}

    	std::cout << "time in milliseconds = " 
    		<< std::chrono::duration_cast<std::chrono::microseconds>(total_time).count()/1000.0f 
            << " ms" << endl;

    }

    return 0;
}
