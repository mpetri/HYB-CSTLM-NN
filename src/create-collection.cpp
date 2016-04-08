#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <iostream>

#include "utils.hpp"
#include "constants.hpp"
#include "collection.hpp"
#include "logging.hpp"

using namespace cstlm;

typedef struct cmdargs {
    std::string input_file;
    std::string collection_dir;
    bool byte_alphabet;
    bool write_corpus;
    uint64_t min_symbol_freq;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -i <input file> -c <coldir> -t <thres> \n", program);
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
    args.write_corpus = false;
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
            args.min_symbol_freq = std::strtoull(optarg, NULL, 10);
            args.write_corpus = true;
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

std::vector<std::string> parse_line(const std::string& line, bool byte)
{
    std::vector<std::string> line_tokens;
    if (byte) {
        for (const auto& chr : line) {
            line_tokens.push_back(std::string(1, chr));
        }
    }
    else {
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

    if (args.byte_alphabet) {
        // handle byte alphabet faster
        LOG(INFO) << "determine symbol frequencies";
        std::array<uint64_t, 256> syms{ 0 };
        {
            std::array<uint8_t, 1024 * 1024> buf{ 0 };
            auto f = std::fopen(args.input_file.c_str(), "r");
            if (!f) {
                LOG(FATAL) << "could not open input file '" << args.input_file << "'";
            }
            size_t bytes_read = 0;
            while ((bytes_read = std::fread(buf.data(), 1, buf.size(), f)) != 0) {
                for (size_t i = 0; i < bytes_read; i++)
                    syms[buf[i]]++;
            }
            std::fclose(f);
        }
        // check that we have at least NUM_SPECIAL_SYMS free bytes
        size_t free_syms = 0;
        size_t init_sigma = 0;
        for (size_t i = 0; i < 256; i++) {
            if (syms[i] == 0)
                free_syms++;
            else
                init_sigma++;
        }
        if (free_syms < NUM_SPECIAL_SYMS) {
            LOG(FATAL) << "not enough free byte symbols in input text to insert special symbols";
        }
        // find the correct mapping for the special symbols
        LOG(INFO) << "map special symbols";
        for (size_t j = 0; j < NUM_SPECIAL_SYMS; j++) {
            if (syms[j] == 0) {
                syms[j] = j;
                LOG(INFO) << "MAP " << j << " -> " << j;
            }
            else {
                for (size_t i = NUM_SPECIAL_SYMS; i < 256; i++) {
                    if (syms[i] == 0) {
                        syms[j] = i;
                        LOG(INFO) << "MAP " << j << " -> " << i;
                        syms[i] = std::numeric_limits<uint64_t>::max();
                        break;
                    }
                }
            }
        }
        size_t num_sentences = 0;
        size_t num_tokens = 0;
        {
            LOG(INFO) << "write parsed text file";
            std::ifstream ifs(args.input_file);
            std::string line;
            auto out = sdsl::write_out_buffer<8>::create(args.collection_dir + "/" + KEY_PREFIX_BYTE + KEY_TEXT);
            out.push_back(EOS_SYM); // file starts with EOS_SYM

            while (std::getline(ifs, line)) {
                out.push_back(PAT_START_SYM); // line starts with PAT_START_SYM
                for (uint8_t chr : line) {
                    if (chr < NUM_SPECIAL_SYMS) {
                        // need to replace sym with other symbol
                        chr = syms[chr];
                    }
                    out.push_back(chr);
                    num_tokens++;
                }
                out.push_back(PAT_END_SYM); // line ends with PAT_END_SYM
                out.push_back(EOS_SYM);
                num_sentences++;
            }
            { // include special 'UNK' sentence to ensure symbol included in CST
                out.push_back(UNKNOWN_SYM);
                out.push_back(EOS_SYM);
            }
            out.push_back(EOF_SYM);
        }

        // write dict
        LOG(INFO) << "write vocab file";
        {
            std::ofstream ofs(args.collection_dir + "/" + KEY_PREFIX_BYTE + KEY_VOCAB);
            // write special symbols
            ofs << "<EOF> 0\n";
            ofs << "<EOS> 1\n";
            ofs << "<UNK> 2\n";
            ofs << "<S> 3\n";
            ofs << "</S> 4\n";
            // write the real vocab
            for (size_t i = 0; i < NUM_SPECIAL_SYMS; i++) {
                if (syms[i] != i)
                    ofs << i << " " << syms[i] << std::endl;
            }
            for (size_t i = NUM_SPECIAL_SYMS; i < 256; i++) {
                if (syms[i] != 0 && syms[i] != std::numeric_limits<uint64_t>::max())
                    ofs << i << " " << i << std::endl;
            }
        }

        LOG(INFO) << "write stats file";
        {
            std::ofstream ofs(args.collection_dir + "/" + KEY_PREFIX_BYTE + KEY_STATS);
            ofs << "initial_vocab_size=" << init_sigma << "\n";
            LOG(INFO) << "initial_vocab_size=" << init_sigma;
            ofs << "num_sentences=" << num_sentences << "\n";
            LOG(INFO) << "num_sentences=" << num_sentences;
            ofs << "num_tokens=" << num_tokens << "\n";
            LOG(INFO) << "num_tokens=" << num_tokens;
            ofs << "raw_size_in_bytes=" << sdsl::util::file_size(args.input_file) << "\n";
            LOG(INFO) << "raw_size_in_bytes=" << sdsl::util::file_size(args.input_file);
            ofs << "min_symbol_freq=" << 0 << "\n";
            LOG(INFO) << "min_symbol_freq=" << 0;
        }
    }
    else {
        /* (1) create dict */
        std::unordered_map<std::string, uint64_t> dict;
        dict.max_load_factor(0.2);
        std::vector<std::pair<uint64_t, std::string> > dict_ids;
        size_t max_id = 0;
        uint64_t initial_sigma = 0;
        uint64_t pruned_sigma = 0;
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
            std::sort(dict_ids.begin(), dict_ids.end(),
                std::greater<std::pair<uint64_t, std::string> >());

            initial_sigma = dict_ids.size();

            LOG(INFO) << "remove low freq (<" << args.min_symbol_freq << ") symbols";
            for (size_t i = 0; i < dict_ids.size(); i++) {
                if (dict_ids[i].first < args.min_symbol_freq) {
                    LOG(INFO) << "initial sigma = " << dict_ids.size();
                    LOG(INFO) << "initial log2(sigma) = "
                              << sdsl::bits::hi(dict_ids.size()) + 1;
                    LOG(INFO) << "new sigma = " << i + 1;
                    LOG(INFO) << "new log2(sigma) = " << sdsl::bits::hi(i + 1) + 1;
                    dict_ids.resize(i + 1);
                    break;
                }
            }
            pruned_sigma = dict_ids.size();

            LOG(INFO) << "create id mapping";
            uint64_t cur_id = NUM_SPECIAL_SYMS;
            for (const auto& did : dict_ids) {
                dict[did.second] = cur_id;
                max_id = cur_id;
                cur_id++;
            }
        }
        uint64_t num_non_freq_syms = 0;
        uint64_t num_sentences = 0;
        uint64_t num_tokens = 0;
        LOG(INFO) << "2nd pass to transform the integers";
        {
            std::ofstream corpus_word;
            if (args.write_corpus) {
                corpus_word.open(args.collection_dir + "/corpus.WORD");
            }

            auto buf = sdsl::write_out_buffer<0>::create(args.collection_dir + "/" + KEY_PREFIX + KEY_TEXT);
            auto int_width = sdsl::bits::hi(max_id) + 1;
            buf.width(int_width);
            std::ifstream ifs(args.input_file);
            std::string line;
            buf.push_back(EOS_SYM); // file starts with EOS_SYM
            while (std::getline(ifs, line)) {
                buf.push_back(PAT_START_SYM); // line starts with PAT_START_SYM
                auto line_tokens = parse_line(line, args.byte_alphabet);
                for (const auto& tok : line_tokens) {
                    auto itr = dict.find(tok);
                    if (itr == dict.end()) {
                        if (args.write_corpus)
                            corpus_word << "<NOT_FREQ> ";
                        buf.push_back(NOT_FREQ_SYM);
                        isreplaced = true;
                        num_non_freq_syms++;
                    }
                    else {
                        if (args.write_corpus)
                            corpus_word << tok + " ";
                        auto num = itr->second;
                        buf.push_back(num);
                    }
                }
                if (args.write_corpus)
                    corpus_word << "\n";
                buf.push_back(PAT_END_SYM); // line ends with PAT_END_SYM
                buf.push_back(EOS_SYM);
                num_sentences++;
            }
            { // include special 'UNK' sentence to ensure symbol included in CST
                buf.push_back(UNKNOWN_SYM);
                if (isreplaced)
                    buf.push_back(NOT_FREQ_SYM);
                buf.push_back(EOS_SYM);
            }
            buf.push_back(EOF_SYM);
            num_tokens = buf.size();
            if (args.write_corpus)
                corpus_word.close();
            LOG(INFO) << "text size = " << buf.size();
            LOG(INFO) << "num_non_freq_syms = " << num_non_freq_syms;
            LOG(INFO) << "non freq percent = "
                      << 100.0 * ((double)num_non_freq_syms / (double)buf.size());
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
            if (isreplaced)
                ofs << "<NOT_FREQ> 5\n";
            // write the real vocab
            uint64_t cur_id = NUM_SPECIAL_SYMS;
            for (const auto& did : dict_ids) {
                ofs << did.second << " " << cur_id << "\n";
                cur_id++;
            }
        }
        LOG(INFO) << "write stats file";
        {
            std::ofstream ofs(args.collection_dir + "/" + KEY_PREFIX + KEY_STATS);
            ofs << "initial_vocab_size=" << initial_sigma << "\n";
            LOG(INFO) << "initial_vocab_size=" << initial_sigma;
            ofs << "pruned_vocab_size=" << pruned_sigma << "\n";
            LOG(INFO) << "pruned_vocab_size=" << pruned_sigma;
            ofs << "num_non_freq_syms=" << num_non_freq_syms << "\n";
            LOG(INFO) << "num_non_freq_syms=" << num_non_freq_syms;
            ofs << "num_sentences=" << num_sentences << "\n";
            LOG(INFO) << "num_sentences=" << num_sentences;
            ofs << "num_tokens=" << num_tokens << "\n";
            LOG(INFO) << "num_tokens=" << num_tokens;
            ofs << "raw_size_in_bytes=" << sdsl::util::file_size(args.input_file) << "\n";
            LOG(INFO) << "raw_size_in_bytes=" << sdsl::util::file_size(args.input_file);
            ofs << "min_symbol_freq=" << args.min_symbol_freq << "\n";
            LOG(INFO) << "min_symbol_freq=" << args.min_symbol_freq;
        }
    }

    return 0;
}
