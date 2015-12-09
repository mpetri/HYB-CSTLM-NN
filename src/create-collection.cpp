#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <iostream>

#include "utils.hpp"
#include "constants.hpp"
#include "collection.hpp"
#include "logging.hpp"

typedef struct cmdargs {
    std::string input_file;
    std::string collection_dir;
    bool byte_alphabet;
    uint64_t min_symbol_freq;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -i <input file> -c <coldir> -t <thres>\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -i <input file>      : the input file.\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -t <threshold>       : min symbol freq.\n");
    fprintf(stdout, "  -1                   : byte parsing.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int op;
    args.input_file = "";
    args.collection_dir = "";
    args.byte_alphabet = false;
    args.min_symbol_freq = 0;
    while ((op = getopt(argc, (char* const*)argv, "i:c:1t:")) != -1) {
        switch (op) {
        case 'i':
            args.input_file = optarg;
            break;
        case 'c':
            args.collection_dir = optarg;
            break;
        case 't':
            args.min_symbol_freq = std::strtoull(optarg,NULL,10);
            break;
        case '1':
            args.byte_alphabet = true;
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

std::vector<std::string>
parse_line(const std::string& line, bool byte)
{
    std::vector<std::string> line_tokens;
    if (byte) {
        for (const auto& chr : line) {
            line_tokens.push_back(std::string(1, chr));
        }
    } else {
        std::istringstream input(line);
        std::string word;
        while (std::getline(input, word, ' ')) {
            line_tokens.push_back(word);
        }
    }
    return line_tokens;
}

int main(int argc, const char* argv[])
{
    log::start_log(argc, argv);
    bool isreplaced = false;

    LOG(INFO) << "parse command line";
    cmdargs_t args = parse_args(argc, argv);
    LOG(INFO) << "collection dir = " << args.collection_dir;
    LOG(INFO) << "input file = " << args.input_file;
    LOG(INFO) << "min_symbol_freq = " << args.min_symbol_freq;
    LOG(INFO) << "byte_alphabet = " << args.byte_alphabet;

    /* create collection dir */
    utils::create_directory(args.collection_dir);

    /* (1) create dict */
    std::unordered_map<std::string, uint64_t> dict;
    dict.max_load_factor(0.2);
    std::vector<std::pair<uint64_t, std::string> > dict_ids;
    size_t max_id = 0;
    {
        LOG(INFO) << "tokenize text";
        std::unordered_map<std::string, uint64_t> tdict;
        tdict.max_load_factor(0.2);
        std::ifstream ifs(args.input_file);
        std::string line;
        while (std::getline(ifs, line)) {
            auto line_tokens = parse_line(line, args.byte_alphabet);
            for (const auto& tok : line_tokens)
                tdict[tok]++;
        }
        LOG(INFO) << "sort by freq";
        for (const auto& did : tdict) {
            dict_ids.emplace_back(did.second, did.first);
        }
        std::sort(dict_ids.begin(), dict_ids.end(), std::greater<std::pair<uint64_t, std::string> >());

        LOG(INFO) << "remove low freq (<"<<args.min_symbol_freq<<") symbols";
        for(size_t i=0;i<dict_ids.size();i++) {
            if(dict_ids[i].first < args.min_symbol_freq) {
                LOG(INFO) << "initial sigma = " << dict_ids.size();
                LOG(INFO) << "initial log2(sigma) = " << sdsl::bits::hi(dict_ids.size()) + 1;
                LOG(INFO) << "new sigma = " << i+1;
                LOG(INFO) << "new log2(sigma) = " << sdsl::bits::hi(i+1) + 1;
                dict_ids.resize(i+1);
                break;
            }
        }

        LOG(INFO) << "create id mapping";
        uint64_t cur_id = NUM_SPECIAL_SYMS;
        for (const auto& did : dict_ids) {
            dict[did.second] = cur_id;
            max_id = cur_id;
            cur_id++;
        }
    }
    LOG(INFO) << "2nd pass to transform the integers";
    {
        auto buf = sdsl::write_out_buffer<0>::create(args.collection_dir + "/" + KEY_PREFIX + KEY_TEXT);
        auto int_width = sdsl::bits::hi(max_id) + 1;
        buf.width(int_width);
        std::ifstream ifs(args.input_file);
        std::string line;
        buf.push_back(EOS_SYM); // file starts with EOS_SYM
        uint64_t num_non_freq_syms = 0;
	bool isreplaced = false;
        while (std::getline(ifs, line)) {
            buf.push_back(PAT_START_SYM); // line starts with PAT_START_SYM
            auto line_tokens = parse_line(line, args.byte_alphabet);
            for (const auto& tok : line_tokens) {
                auto itr = dict.find(tok);
                if(itr == dict.end()) {
                    buf.push_back(NOT_FREQ_SYM);
		    isreplaced = true;
                    num_non_freq_syms++;
                } else {
                    auto num = itr->second;
                    buf.push_back(num);
                }
            }
            buf.push_back(PAT_END_SYM); // line ends with PAT_END_SYM
            buf.push_back(EOS_SYM);
        }
        { // include special 'UNK' sentence to ensure symbol included in CST
            buf.push_back(UNKNOWN_SYM);
	    if(isreplaced)
               buf.push_back(NOT_FREQ_SYM);
            buf.push_back(EOS_SYM);
        }
        buf.push_back(EOF_SYM);
        LOG(INFO) << "text size = " << buf.size();
        LOG(INFO) << "num_non_freq_syms = " << num_non_freq_syms;
        LOG(INFO) << "non freq percent = " << 100.0 * ( (double) num_non_freq_syms / (double) buf.size());
    }
    LOG(INFO) << "write vocab file";
    {
        std::ofstream ofs(args.collection_dir + "/" + KEY_PREFIX + KEY_VOCAB);
        // write special symbols
        ofs << "<EOF> 0\n";
        ofs << "<EOS> 1\n";
        ofs << "<UNK> 2\n";
        ofs << "<S> 3\n";
        ofs << "</S> 4\n";
	if(isreplaced)
           ofs << "<NOT_FREQ> 5\n";
        // write the real vocab
        uint64_t cur_id = NUM_SPECIAL_SYMS;
        for (const auto& did : dict_ids) {
            ofs << did.second << " " << cur_id << "\n";
            cur_id++;
        }
    }
    return 0;
}
