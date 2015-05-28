#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <iostream>

#include "utils.hpp"
#include "constants.hpp"
#include "collection.hpp"
#include "vocab_uncompressed.hpp"

typedef struct cmdargs {
    std::string input_file;
    std::string collection_dir;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -i <input file>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -i <input file>  : the input file.\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.input_file = "";
    args.collection_dir = "";
    while ((op = getopt(argc, (char* const*)argv, "i:c:")) != -1) {
        switch (op) {
        case 'i':
            args.input_file = optarg;
            break;
        case 'c':
            args.collection_dir = optarg;
            break;
        }
    }
    if (args.collection_dir == "" || args.input_file == "") {
        std::cerr << "Missing command line parameters.\n";
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

    /* load vocab */
    vocab_uncompressed vocab(col);

    /* parse query file */
    std::ifstream ifs(args.input_file);
    std::string line;
    size_t num_line = 1;
    while( std::getline(ifs,line) ) {
        std::istringstream input(line);
        std::string word;
        bool contains_ukn = false;
        while( std::getline(input,word,' ') ) {
            try {
                vocab.token2id(word);
            } catch(const std::runtime_error& e) {
                // LOG(ERROR) << "line=" << num_line << " cannot find word '" << word << "'";
                contains_ukn = true;
            }
        }
        if(!contains_ukn) {
            std::cout << line << "\n";
        }
        num_line++;
    }

    return 0;
}
