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

#include <fstream>
#include <iostream>

using namespace std;
using namespace dynet;
using namespace dynet::expr;
using namespace cstlm;

typedef struct cmdargs {
	std::string collection_dir;
	std::string word2vec_file;
	bool		use_mkn;
} cmdargs_t;

struct word2vec_embeddings {
};

void print_usage(const char* program)
{
	fprintf(stdout, "%s -c <collection dir>\n", program);
	fprintf(stdout, "where\n");
	fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
	fprintf(stdout, "  -m                   : use modified kneser ney.\n");
	fprintf(stdout, "  -t <threads>         : limit the number of threads.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
	cmdargs_t args;
	int		  op;
	args.collection_dir = "";
	args.use_mkn		= false;
	while ((op = getopt(argc, (char* const*)argv, "c:dmt:")) != -1) {
		switch (op) {
			case 'c':
				args.collection_dir = optarg;
				break;
			case 'm':
				args.use_mkn = true;
				break;
			case 't':
				num_cstlm_threads = std::atoi(optarg);
				break;
		}
	}
	if (args.collection_dir == "") {
		LOG(FATAL) << "Missing command line parameters.";
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	return args;
}

template <class t_idx>
t_idx create_and_store(collection& col, bool use_mkn)
{
	using clock = std::chrono::high_resolution_clock;
	auto  start = clock::now();
	t_idx idx(col, use_mkn);
	auto  stop = clock::now();
	LOG(INFO) << "index construction in (s): "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() /
				 1000.0f;
	auto output_file = col.path + "/index/index-" + sdsl::util::class_to_hash(idx) + ".sdsl";
	std::ofstream ofs(output_file);
	if (ofs.is_open()) {
		LOG(INFO) << "writing index to file : " << output_file;
		auto bytes = sdsl::serialize(idx, ofs);
		LOG(INFO) << "index size : " << bytes / (1024 * 1024) << " MB";
		sdsl::write_structure<sdsl::HTML_FORMAT>(idx, output_file + ".html");
	} else {
		LOG(FATAL) << "cannot write index to file : " << output_file;
	}
	return idx;
}

word2vec::embeddings load_or_create_word2vec_embeddings(collection& col)
{
	auto embeddings = word2vec::builder{}
					  .vector_size(200)
					  .window_size(5)
					  .sample_threadhold(1e-5)
					  .num_negative_samples(5)
					  .num_threads(cstlm::num_cstlm_threads)
					  .num_iterations(5)
					  .min_freq_threshold(5)
					  .start_learning_rate(0.025)
					  .batch_size(11)
					  .use_cbow(false)
					  .train_or_load(col);
	return embeddings;
}


int main(int argc, char** argv)
{
	dynet::initialize(argc, argv);
	enable_logging = true;

	/* parse command line */
	cmdargs_t args = parse_args(argc, (const char**)argv);

	/* (1) parse collection directory and create CSTLM index */
	collection col(args.collection_dir);

	/* (2) load the word2vec embeddings */
	auto word_embeddings = load_or_create_word2vec_embeddings(col);

	/* (3) */
	auto cstlm = create_and_store<wordlm>(col, args.use_mkn);

	ComputationGraph cg;

	return 0;
}
