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

template<class t_idx>
std::vector<std::pair<uint64_t,uint64_t>>
run_query_knm(const t_idx& idx,const std::vector<uint64_t>& tokens)
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
    		auto res = run_query_knm(idx,tokens);
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
