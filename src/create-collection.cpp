#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <iostream>

#include "utils.hpp"
#include "constants.hpp"
#include "collection.hpp"

typedef struct cmdargs {
    std::string input_file;
    std::string list_file;
    std::string collection_dir;
    bool is_bytes;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -i <input file>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -i <input file>  : the input file.\n");
    fprintf(stdout, "  -l <input file file> : file containing list of file names.\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -b : treat as stream of bytes.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.input_file = "";
    args.collection_dir = "";
    args.list_file = "";
    args.is_bytes = false;
    while ((op = getopt(argc, (char* const*)argv, "i:c:l:b")) != -1) {
        switch (op) {
        case 'i':
            args.input_file = optarg;
            break;
        case 'c':
            args.collection_dir = optarg;
            break;
        case 'l':
            args.list_file = optarg;
            break;
        case 'b':
            args.is_bytes = true;
            break;
        }
    }
    if (args.collection_dir == "" || (args.input_file == "" && args.list_file == "")) {
        std::cerr << "Missing command line parameters.\n";
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return args;
}

int main(int argc, const char* argv[])
{
    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    std::vector<std::string> input_filenames;
    if (utils::file_exists(args.list_file)) {
        std::ifstream ifile(args.list_file);
        std::string line;
        while (std::getline(ifile, line)) {
            input_filenames.push_back(line);
            assert(utils::file_exists(input_filenames.back()));
        }
    } else {
        input_filenames.push_back(args.input_file);
        assert(utils::file_exists(input_filenames.back()));
    }

    /* parse file and output to sdsl format */
    /* create sdsl input file on disk */
    std::string output_file = args.collection_dir + +"/" + KEY_PREFIX + KEY_TEXT;
    std::cout << "writing to output file '" << output_file << "'" << std::endl;
    {
        sdsl::int_vector<> tmp;
        std::ofstream ofs(output_file);
        sdsl::serialize(tmp, ofs);
    }
    sdsl::int_vector_mapper<0, std::ios_base::out | std::ios_base::in> sdsl_input(output_file);
    std::ifstream ifile(args.input_file);

    std::unordered_map<std::string, uint64_t> vocab;

    sdsl_input.push_back(EOS_SYM); // TODO added
    for (auto input_file: input_filenames) {
        std::cout << "reading input file '" << input_file << "'" << std::endl;
        std::string line;
        uint64_t num;
        std::ifstream ifile(input_file);
        while (std::getline(ifile, line)) {
            sdsl_input.push_back(PAT_START_SYM); // TODO added
            if (!args.is_bytes) {
                std::istringstream iss(line);
                std::string word;
                while (std::getline(iss, word, ' ')) {
                    //uint64_t num = std::stoull(word);
                    auto it = vocab.find(word);
                    if (it != vocab.end())
                        num = it->second;
                    else {
                        num = vocab.size() + NUM_SPECIAL_SYMS;
                        vocab[word] = num;
                    }
                    sdsl_input.push_back(num);
                }
            } else { // process as bytes
                std::istringstream iss(line);
                char ch;
                while (iss >> ch) {
                    num = ch + NUM_SPECIAL_SYMS;
                    sdsl_input.push_back(num);
                }
            }
            sdsl_input.push_back(PAT_END_SYM); // TODO added
            sdsl_input.push_back(EOS_SYM);
        }
    }
    sdsl_input.push_back(EOF_SYM);
    sdsl::util::bit_compress(sdsl_input);

    if (args.is_bytes) {
        for (int i = 0; i < 256; ++i) {
            std::ostringstream oss;
            oss << i;
            vocab[oss.str()] = i + NUM_SPECIAL_SYMS;
        }
    }

    std::string vocab_file = args.collection_dir + +"/" + KEY_PREFIX + KEY_VOCAB;
    std::cout << "writing to vocab file '" << vocab_file << "'" << std::endl;
    std::ofstream ofs(vocab_file);
    ofs << "<EOF> 0\n";
    ofs << "<EOS> 1\n";
    ofs << "<UNK> 2\n";
    ofs << "<S> 3\n";
    ofs << "</S> 4\n";
    for (auto wiit = vocab.begin(); wiit != vocab.end(); ++ wiit) 
        ofs << wiit->first << ' ' << wiit->second << "\n";

    return 0;
}
