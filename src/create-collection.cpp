#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <iostream>

#include "utils.hpp"
#include "collection.hpp"

typedef struct cmdargs {
    std::string input_file;
    std::string collection_dir;
} cmdargs_t;

void
print_usage(const char* program)
{
    fprintf(stdout,"%s -i <input file>\n",program);
    fprintf(stdout,"where\n");
    fprintf(stdout,"  -i <input file>  : the input file.\n");
    fprintf(stdout,"  -c <collection dir>  : the collection dir.\n");
};

cmdargs_t
parse_args(int argc,const char* argv[])
{
    cmdargs_t args;
    int op;
    args.input_file = "";
    args.collection_dir = "";
    while ((op=getopt(argc,(char* const*)argv,"i:c:")) != -1) {
        switch (op) {
            case 'i':
                args.input_file = optarg;
                break;
            case 'c':
                args.collection_dir = optarg;
                break;
        }
    }
    if (args.collection_dir==""||args.input_file=="") {
        std::cerr << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}


int main(int argc,const char* argv[])
{
    /* parse command line */
    cmdargs_t args = parse_args(argc,argv);

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    /* parse file and output to sdsl format */
    if(utils::file_exists(args.input_file)) {
    	/* create sdsl input file on disk */
    	std::string output_file = args.collection_dir + +"/"+KEY_PREFIX+KEY_TEXT;
    	std::cout << "writing to output file '" << output_file << "'" << std::endl;
    	{
    		sdsl::int_vector<> tmp;
    		std::ofstream ofs(output_file);
    		sdsl::serialize(tmp,ofs);
    	}
    	sdsl::int_vector_mapper<0,std::ios_base::out|std::ios_base::in> sdsl_input(output_file);
    	std::ifstream ifile(args.input_file);
    	std::cout << "reading input file '" << args.input_file << "'" << std::endl;
    	std::string line;
    	while (std::getline(ifile,line)) {
    		std::istringstream iss(line);
    		std::string word;
    		while (std::getline(iss, word, ' ')) {
    			uint64_t num = std::stoull(word);
    			sdsl_input.push_back(num);
    		}
    		sdsl_input.push_back(1ULL); // EOL
    	}
    	sdsl_input.push_back(0ULL); // EOL
    	sdsl::util::bit_compress(sdsl_input);
    } else {
    	std::cerr << "input file does not exist." << std::endl;
    }

    return 0;
}