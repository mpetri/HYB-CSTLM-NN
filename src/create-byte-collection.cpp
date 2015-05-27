#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <iostream>

#include "utils.hpp"
#include "constants.hpp"
#include "collection.hpp"

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
    /* parse command line */
    cmdargs_t args = parse_args(argc, argv);

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    /* (1) create dict */
    std::unordered_map<char,uint64_t> dict;
    std::vector<std::pair<uint64_t,char>> dict_ids;
    size_t max_id = 0;
    {
        std::unordered_map<char,uint64_t> tdict;
        std::ifstream ifs(args.input_file);
        std::string line;
        while( std::getline(ifs,line) ) {
            for(const auto& chr : line) {
                tdict[chr]++;
            }
        }
        /* sort by freq */
        for(const auto& did : tdict) {
            dict_ids.emplace_back(did.second,did.first);
        }
        std::sort(dict_ids.begin(),dict_ids.end(),std::greater<std::pair<uint64_t,char>>());
        /* create id mapping */
        uint64_t cur_id = NUM_SPECIAL_SYMS;
        for(const auto& did: dict_ids) {
            dict[did.second] = cur_id;
            max_id = cur_id;
            cur_id++;
        }
    }
    /* (2) 2nd pass to transform the integers */
    {
        auto buf = sdsl::write_out_buffer<0>::create(args.collection_dir+"/"+ KEY_PREFIX + KEY_TEXT);
        auto int_width = sdsl::bits::hi(max_id)+1;
        buf.width(int_width);
        std::ifstream ifs(args.input_file);
        std::string line;
        buf.push_back(EOS_SYM); // file starts with EOS_SYM
        while( std::getline(ifs,line) ) {
            std::istringstream input(line);
            std::string word;
            buf.push_back(PAT_START_SYM); // line starts with PAT_START_SYM
            for(const auto& chr : line) {
                auto itr = dict.find(chr);
                auto num = itr->second;
                buf.push_back(num);
            }
            buf.push_back(PAT_END_SYM); // line ends with PAT_END_SYM
        }
        buf.push_back(EOF_SYM);
        // sdsl::util::bit_compress(buf);
    }
    /* (3) write vocab file */ 
    {
        std::ofstream ofs(args.collection_dir+"/"+ KEY_PREFIX + KEY_VOCAB);
        uint64_t cur_id = NUM_SPECIAL_SYMS;
        for(const auto& did: dict_ids) {
            ofs << did.second << " " << cur_id << "\n";
            cur_id++;
        }
    }

    return 0;
}
