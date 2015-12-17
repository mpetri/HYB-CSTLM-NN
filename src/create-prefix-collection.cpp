#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"

#include "utils.hpp"
#include "index_types.hpp"
#include "logging.hpp"
#include "mem_monitor.hpp"

typedef struct cmdargs {
    std::string collection_dir;
    std::string prefix_collection_dir;
    double percent;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c <collection dir> -p <prefix collection> -s <percent>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -p <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -s                   : size of the prefix in percent.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.collection_dir = "";
    args.prefix_collection_dir = "";
    args.percent = 1.0;
    while ((op = getopt(argc, (char* const*)argv, "c:p:s:")) != -1) {
        switch (op) {
        case 'c':
            args.collection_dir = optarg;
            break;
        case 'p':
            args.prefix_collection_dir = optarg;
            break;
        case 's':
            args.percent = std::atof(optarg);
            break;
        }
    }
    if (args.collection_dir == "" || args.prefix_collection_dir == "") {
        LOG(FATAL) << "Missing command line parameters.";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

int main(int argc, const char* argv[])
{
    log::start_log(argc, argv);

    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* parse collection directory */
    collection col(args.collection_dir);

    {
    	sdsl::read_only_mapped_buffer<0> text(col.file_map[KEY_TEXT]);
    	uint64_t num_sentences = 0;
    	for(const auto& sym : text) {
    		if(sym == EOS_SYM) num_sentences++;
    	}
    	auto new_num_sentences = num_sentences * args.percent;

	    /* create collection dir */
	    utils::create_directory(args.prefix_collection_dir);

    	auto new_text = sdsl::write_out_buffer<0>::create(args.prefix_collection_dir + "/" + KEY_PREFIX + KEY_TEXT);
    	new_text.width(text.width());

    	uint64_t processed_sentences = 0;
    	for(const auto& sym : text) {
    		if(sym == EOS_SYM) processed_sentences++;
    		new_text.push_back(sym);
    		if(processed_sentences == new_num_sentences) {
    			break;
    		}
    	}
    	new_text.push_back(EOF_SYM);
    }
    {
    	std::ifstream ifs(args.collection_dir + "/" + KEY_PREFIX + KEY_VOCAB);
    	std::ofstream ofs(args.prefix_collection_dir + "/" + KEY_PREFIX + KEY_VOCAB);
    	std::string line;
    	while(std::getline(ifs,line)) {
    		ofs << line << "\n";
    	}
    }

    return 0;
}
