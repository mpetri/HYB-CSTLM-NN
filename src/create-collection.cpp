#include <iostream>
#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>

#include "collection.hpp"
#include "constants.hpp"
#include "logging.hpp"
#include "utils.hpp"

using namespace cstlm;

typedef struct cmdargs {
	std::string big_input_file;
	std::string small_input_file;
	std::string collection_dir;
} cmdargs_t;

void print_usage(const char* program)
{
	fprintf(
	stdout, "%s -b <big input file> -s <small input file> -c <coldir> -t <thres> \n", program);
	fprintf(stdout, "where\n");
	fprintf(stdout, "  -b <big input file>    : the big input file.\n");
	fprintf(stdout, "  -s <small input file>  : the small input file.\n");
	fprintf(stdout, "  -c <collection dir>    : the collection dir.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
	cmdargs_t args;
	int		  op;
	args.big_input_file   = "";
	args.small_input_file = "";
	args.collection_dir   = "";
	while ((op = getopt(argc, (char* const*)argv, "b:s:c:")) != -1) {
		switch (op) {
			case 'b':
				args.big_input_file = optarg;
				break;
			case 's':
				args.small_input_file = optarg;
				break;
			case 'c':
				args.collection_dir = optarg;
				break;
		}
	}
	if (args.collection_dir == "" || args.big_input_file == "" || args.small_input_file == "") {
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
	} else {
		std::istringstream input(line);
		std::string		   word;
		while (std::getline(input, word, ' ')) {
			line_tokens.push_back(word);
		}
	}
	return line_tokens;
}

int main(int argc, const char* argv[])
{
	enable_logging  = true;
	bool isreplaced = false;

	LOG(INFO) << "parse command line";
	cmdargs_t args = parse_args(argc, argv);
	LOG(INFO) << "collection dir = " << args.collection_dir;
	LOG(INFO) << "big input file = " << args.big_input_file;
	LOG(INFO) << "small input file = " << args.small_input_file;

	/* create collection dir */
	utils::create_directory(args.collection_dir);

	/* (1) create dict */
	std::unordered_map<std::string, uint64_t> dict;
	dict.max_load_factor(0.2);
	std::vector<std::pair<uint64_t, std::string>> dict_ids;
	size_t   max_id = 0;
	uint64_t sigma  = 0;
	{
		LOG(INFO) << "tokenize texts";
		std::unordered_map<std::string, uint64_t> tdict;
		tdict.max_load_factor(0.2);

		{ // (1a) big file
			LOG(INFO) << "tokenize big text (" << args.big_input_file << ")";
			std::ifstream ifs_big(args.big_input_file);
			std::string   line;
			while (std::getline(ifs_big, line)) {
				auto line_tokens = parse_line(line, false);
				for (const auto& tok : line_tokens)
					tdict[tok]++;
			}
		}

		{ // (1b) small file
			LOG(INFO) << "tokenize small text (" << args.big_input_file << ")";
			std::ifstream ifs_small(args.small_input_file);
			std::string   line;
			while (std::getline(ifs_small, line)) {
				auto line_tokens = parse_line(line, false);
				for (const auto& tok : line_tokens)
					tdict[tok]++;
			}
		}

		LOG(INFO) << "sort dict by freq";
		for (const auto& did : tdict) {
			dict_ids.emplace_back(did.second, did.first);
		}

		std::sort(
		dict_ids.begin(), dict_ids.end(), std::greater<std::pair<uint64_t, std::string>>());

		sigma = dict_ids.size();

		LOG(INFO) << "create id mapping";
		uint64_t cur_id = NUM_SPECIAL_SYMS;
		for (const auto& did : dict_ids) {
			dict[did.second] = cur_id;
			max_id			 = cur_id;
			cur_id++;
		}
	}


	uint64_t big_num_sentences = 0;
	uint64_t big_num_tokens	= 0;
	LOG(INFO) << "2nd pass to transform the integers for big file";
	{
		auto int_width = sdsl::bits::hi(max_id) + 1;
		auto buf = sdsl::int_vector_buffer<0>(args.collection_dir + "/" + KEY_PREFIX + KEY_TEXT,
											  std::ios::out,
											  1024 * 1024 * 128,
											  int_width);
		std::ifstream ifs(args.big_input_file);
		std::string   line;
		buf.push_back(EOS_SYM); // file starts with EOS_SYM
		while (std::getline(ifs, line)) {
			buf.push_back(PAT_START_SYM); // line starts with PAT_START_SYM
			auto line_tokens = parse_line(line, false);
			for (const auto& tok : line_tokens) {
				auto itr = dict.find(tok);
				if (itr == dict.end()) {
					LOG(ERROR) << "can't find symbol '" << tok << "' in dict.";
				} else {
					auto num = itr->second;
					buf.push_back(num);
				}
			}
			buf.push_back(PAT_END_SYM); // line ends with PAT_END_SYM
			buf.push_back(EOS_SYM);
			big_num_sentences++;
		}
		{ // include special 'UNK' sentence to ensure symbol included in CST
			buf.push_back(UNKNOWN_SYM);
			buf.push_back(EOS_SYM);
		}
		buf.push_back(EOF_SYM);
		big_num_tokens = buf.size();
		LOG(INFO) << "big text size = " << buf.size();
	}

	uint64_t small_num_sentences = 0;
	uint64_t small_num_tokens	= 0;
	LOG(INFO) << "2nd pass to transform the integers for small file";
	{
		auto int_width = sdsl::bits::hi(max_id) + 1;
		auto buf =
		sdsl::int_vector_buffer<0>(args.collection_dir + "/" + KEY_PREFIX + KEY_SMALLTEXT,
								   std::ios::out,
								   1024 * 1024 * 128,
								   int_width);
		std::ifstream ifs(args.small_input_file);
		std::string   line;
		buf.push_back(EOS_SYM); // file starts with EOS_SYM
		while (std::getline(ifs, line)) {
			buf.push_back(PAT_START_SYM); // line starts with PAT_START_SYM
			auto line_tokens = parse_line(line, false);
			for (const auto& tok : line_tokens) {
				auto itr = dict.find(tok);
				if (itr == dict.end()) {
					LOG(ERROR) << "can't find symbol '" << tok << "' in dict.";
				} else {
					auto num = itr->second;
					buf.push_back(num);
				}
			}
			buf.push_back(PAT_END_SYM); // line ends with PAT_END_SYM
			buf.push_back(EOS_SYM);
			small_num_sentences++;
		}
		{ // include special 'UNK' sentence to ensure symbol included in CST
			buf.push_back(UNKNOWN_SYM);
			if (isreplaced) buf.push_back(NOT_FREQ_SYM);
			buf.push_back(EOS_SYM);
		}
		buf.push_back(EOF_SYM);
		small_num_tokens = buf.size();
		LOG(INFO) << "small text size = " << buf.size();
	}

	LOG(INFO) << "write vocab file";
	{
		std::ofstream ofs(args.collection_dir + "/" + KEY_PREFIX + KEY_VOCAB);
		// write special symbols
		ofs << "<EOF> 0 1\n";
		ofs << "<EOS> 1 1\n";
		ofs << "<UNK> 2 1\n";
		ofs << "<S> 3 1\n";
		ofs << "</S> 4 1\n";
		// write the real vocab
		uint64_t cur_id = NUM_SPECIAL_SYMS;
		for (const auto& did : dict_ids) {
			ofs << did.second << " " << cur_id << " " << did.first << "\n";
			cur_id++;
		}
	}
	LOG(INFO) << "write stats file";
	{
		std::ofstream ofs(args.collection_dir + "/" + KEY_PREFIX + KEY_STATS);
		ofs << "vocab_size=" << sigma << "\n";
		LOG(INFO) << "vocab_size=" << sigma;
		ofs << "big_num_sentences=" << big_num_sentences << "\n";
		LOG(INFO) << "big_num_sentences=" << big_num_sentences;
		ofs << "big_num_tokens=" << big_num_tokens << "\n";
		LOG(INFO) << "big_num_tokens=" << big_num_tokens;
		ofs << "big_raw_size_in_bytes=" << sdsl::util::file_size(args.big_input_file) << "\n";
		LOG(INFO) << "big_raw_size_in_bytes=" << sdsl::util::file_size(args.big_input_file);
		ofs << "small_num_sentences=" << small_num_sentences << "\n";
		LOG(INFO) << "small_num_sentences=" << small_num_sentences;
		ofs << "small_num_tokens=" << small_num_tokens << "\n";
		LOG(INFO) << "small_num_tokens=" << small_num_tokens;
		ofs << "small_raw_size_in_bytes=" << sdsl::util::file_size(args.small_input_file) << "\n";
		LOG(INFO) << "small_raw_size_in_bytes=" << sdsl::util::file_size(args.small_input_file);
	}

	return 0;
}
