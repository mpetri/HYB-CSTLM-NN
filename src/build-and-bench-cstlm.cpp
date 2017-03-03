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

#include "knm.hpp"
#include "nn/nnconstants.hpp"

#include <fstream>
#include <iostream>


using namespace std;
using namespace dynet;
using namespace dynet::expr;
using namespace cstlm;

typedef struct cmdargs {
    std::string collection_dir;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c -t\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -t <threads>         : limit the number of threads.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int       op;
    args.collection_dir = "";
    while ((op = getopt(argc, (char* const*)argv, "c:t:T:")) != -1) {
        switch (op) {
            case 'c':
                args.collection_dir = optarg;
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
t_idx load_or_create_cstlm(collection& col)
{
    t_idx idx;
    auto  output_file =
    col.file_map[KEY_CSTLM_TEXT] + "-cstlm-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if (cstlm::utils::file_exists(output_file)) {
        LOG(INFO) << "CSTLM loading cstlm index from file : " << output_file;
        std::ifstream ifs(output_file);
        idx.load(ifs);
        idx.print_params(true, 10);
        return idx;
    }
    using clock = std::chrono::high_resolution_clock;
    auto start  = clock::now();
    idx         = t_idx(col, true);
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
    idx.print_params(true, 10);
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
std::vector<std::vector<word_token>> load_and_parse_file(collection& col, const t_idx& index)
{
    auto test_file = col.file_map[KEY_TEST];
    LOG(INFO) << "parsing test sentences from file " << test_file;
    auto filtered_vocab =
    index.vocab.filter(col.file_map[KEY_SMALL_TEXT], nnlm::constants::VOCAB_THRESHOLD);
    auto   sentences = sentence_parser::parse_from_raw(test_file, index.vocab, filtered_vocab);
    size_t tokens    = 0;
    for (const auto& s : sentences)
        tokens += s.size();
    LOG(INFO) << "found " << sentences.size() << " sentences (" << tokens << " tokens)";
    return sentences;
}

template <class t_idx>
void evaluate_sentences(std::vector<std::vector<word_token>>& sentences,
                        const t_idx&                          index,
                        size_t                                order)
{
    double   logprobs = 0;
    uint64_t M          = 0;
    uint64_t OOV        = 0;
    uint64_t sent_size  = 0;
    for (const auto& sentence : sentences) {
	for(size_t i=0;i<sentence.size();i++) {
		if(sentence[i].big_id == UNKNOWN_SYM /* && sentence[i].small_id == UNKNOWN_SYM*/ ) OOV++;
	}
        auto eval_res = sentence_logprob_kneser_ney2(index, sentence, M, order, true, false);
        logprobs += eval_res.logprob;
        M += eval_res.tokens;
	sent_size += sentence.size();
    }
    double perplexity = exp( -logprobs / M );
    LOG(INFO) << "CSTLM ORDER: " << order << " PPLX = " << std::setprecision(10) << perplexity
              << " (logprob = " << logprobs 
	      << ";predicted tokens = " << M 
	      << ";OOV = " << OOV
              << ";#W = " << sent_size << ")";
}


int main(int argc, char** argv)
{
    //dynet::initialize(argc, argv);
    enable_logging = true;

    /* parse command line */
    cmdargs_t args = parse_args(argc, (const char**)argv);

    /* (1) parse collection directory and create CSTLM index */
    collection col(args.collection_dir);
    col.file_map[KEY_CSTLM_TEXT] = col.file_map[KEY_COMBINED_TEXT];

    /* (2) create the cstlm model */
    auto cstlm = load_or_create_cstlm<wordlm>(col);

    /* (3) parse test file */
    auto test_sentences = load_and_parse_file(col, cstlm);

    /* (4) evaluate sentences */
    for (size_t i = 2; i < 20; i++) {
        evaluate_sentences(test_sentences, cstlm, i);
    }

    return 0;
}
