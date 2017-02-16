#include "sdsl/suffix_arrays.hpp"
#include "sdsl/suffix_trees.hpp"

#include "index_types.hpp"
#include "logging.hpp"
#include "utils.hpp"

#include "mem_monitor.hpp"

#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/gpu-ops.h"
#include "dynet/nodes.h"
#include "dynet/training.h"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "word2vec.hpp"
#include "hyblm.hpp"

#include "knm.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;
using namespace cstlm;

typedef struct cmdargs {
	std::string collection_dir;
	std::string test_file;
	bool		use_mkn;
} cmdargs_t;

void print_usage(const char* program)
{
	fprintf(stdout, "%s -c -t -T\n", program);
	fprintf(stdout, "where\n");
	fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
	fprintf(stdout, "  -t <threads>         : limit the number of threads.\n");
	fprintf(stdout, "  -T <test file>       : the location of the test file.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
	cmdargs_t args;
	int		  op;
	args.collection_dir = "";
	args.use_mkn		= true;
	args.test_file		= "";
	while ((op = getopt(argc, (char* const*)argv, "c:t:T:")) != -1) {
		switch (op) {
			case 'c':
				args.collection_dir = optarg;
				break;
			case 't':
				num_cstlm_threads = std::atoi(optarg);
				break;
			case 'T':
				args.test_file = optarg;
				break;
		}
	}
	if (args.collection_dir == "" || args.test_file == "") {
		LOG(FATAL) << "Missing command line parameters.";
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	return args;
}

template <class t_idx>
t_idx load_or_create_cstlm(collection& col, bool use_mkn)
{
	t_idx idx;
	auto  output_file = col.path + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
	if (cstlm::utils::file_exists(output_file)) {
		LOG(INFO) << "CSTLM loading cstlm index from file : " << output_file;
		std::ifstream ifs(output_file);
		idx.load(ifs);
        idx.print_params(true,10);
		return idx;
	}
	using clock = std::chrono::high_resolution_clock;
	auto start  = clock::now();
	idx			= t_idx(col, use_mkn);
	auto stop   = clock::now();
	LOG(INFO) << "CSTLM index construction in (s): "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() /
				 1000.0f;

	std::ofstream ofs(output_file);
	if (ofs.is_open()) {
		LOG(INFO) << "CSTLM writing index to file : " << output_file;
		auto bytes = sdsl::serialize(idx, ofs);
		LOG(INFO) << "CSTLM index size : " << bytes / (1024 * 1024) << " MB";
		sdsl::write_structure<sdsl::HTML_FORMAT>(idx, output_file + ".html");
	} else {
		LOG(FATAL) << "CSTLM cannot write index to file : " << output_file;
	}
    idx.print_params(true,10);
	return idx;
}

template <class t_idx>
std::string sentence_to_str(std::vector<uint32_t> sentence, const t_idx& index)
{
	std::string str = "[";
	for (size_t i = 0; i < sentence.size() - 1; i++) {
		auto id_tok  = sentence[i];
		auto str_tok = index.vocab.id2token(id_tok);
		str += str_tok + ",";
	}
	auto str_tok = index.vocab.id2token(sentence.back());
	str += str_tok + "]";
	return str;
}

template <class t_idx>
std::vector<std::vector<uint32_t>> load_and_parse_file(std::string file_name, const t_idx& index)
{
	std::vector<std::vector<uint32_t>> sentences;
	std::ifstream					   ifile(file_name);
	LOG(INFO) << "reading input file '" << file_name << "'";
	std::string line;
	while (std::getline(ifile, line)) {
		auto				  line_tokens = utils::parse_line(line, false);
		std::vector<uint32_t> tokens;
		tokens.push_back(PAT_START_SYM);
		for (const auto& token : line_tokens) {
			auto num = index.vocab.token2id(token, UNKNOWN_SYM);
			tokens.push_back(num);
		}
		tokens.push_back(PAT_END_SYM);
		sentences.push_back(tokens);
	}
	LOG(INFO) << "found " << sentences.size() << " sentences";
	return sentences;
}

template <class t_idx>
void evaluate_sentences(std::vector<std::vector<uint32_t>>& sentences,
						const t_idx&						index,
						size_t								order)
{
	double   perplexity = 0;
	uint64_t M			= 0;
    uint64_t cur = 0;
	for (auto sentence : sentences) 
	{
		uint64_t plen = sentence.size();
		M += plen- 1; // do not count <s>
		double sentenceprob = sentence_logprob_kneser_ney(index, sentence, M, order, true, false);
        double sperplexity = pow(10, -(1.0 / (double)plen-1) * sentenceprob);
		//LOG(INFO) << "S(" << ++cur << ") = " << sentence_to_str(sentence, index) << " PPLX = " <<  sperplexity;
		perplexity += sentenceprob;
	}
	perplexity = perplexity / M;
	LOG(INFO) << "CSTLM ORDER: " << order << " PPLX = " << std::setprecision(10)
			  << pow(10, -perplexity);
}


int main(int argc, char** argv)
{
	dynet::initialize(argc, argv);
	enable_logging = true;

	/* parse command line */
	cmdargs_t args = parse_args(argc, (const char**)argv);

	/* (1) parse collection directory and create CSTLM index */
	collection col(args.collection_dir);
	col.file_map[KEY_CSTLM_TEXT] = col.file_map[KEY_COMBINED_TEXT];

	/* (2) create the cstlm model */
	auto cstlm = load_or_create_cstlm<wordlm>(col, args.use_mkn);

	/* (3) parse test file */
	auto test_sentences = load_and_parse_file(args.test_file, cstlm);

	/* (4) evaluate sentences */
	for (size_t i = 2; i < 20; i++) {
		evaluate_sentences(test_sentences, cstlm, i);
	}

	return 0;
}
