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
#include "rnnlm.hpp"
#include "nn/nnconstants.hpp"

#include "knm.hpp"

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
    while ((op = getopt(argc, (char* const*)argv, "c:t:")) != -1) {
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
std::string sentence_to_str(std::vector<uint32_t> sentence, const t_idx& index)
{
    std::string str = "[";
    for (size_t i = 0; i < sentence.size() - 1; i++) {
        auto id_tok  = sentence[i];
        auto str_tok = index.filtered_vocab.id2token(id_tok);
        str += str_tok + ",";
    }
    auto str_tok = index.filtered_vocab.id2token(sentence.back());
    str += str_tok + "]";
    return str;
}

template <class t_idx>
std::vector<std::vector<word_token>> load_and_parse_file(std::string file_name, const t_idx& index)
{
    auto sentences = index.parse_raw_sentences(file_name);
    size_t tokens = 0;
    for(const auto& s : sentences) tokens += s.size();
    LOG(INFO) << "found " << sentences.size() << " sentences (" << tokens << " tokens)";
    return sentences;
}

void evaluate_sentences(std::vector<std::vector<word_token>>& sentences, rnnlm::LM& rnn_lm)
{
    double perplexity          = 0;
    double num_words_predicted = 0;
    for (const auto& sentence : sentences) {
        auto eval_res = rnn_lm.evaluate_sentence_logprob(sentence);
        num_words_predicted += eval_res.tokens;
        perplexity += eval_res.logprob;
    }
    LOG(INFO) << "RNNLM PPLX = " << std::setprecision(10) << exp(perplexity/num_words_predicted) 
	      << " (predicted tokens = " << num_words_predicted << ")";
}


word2vec::embeddings load_or_create_word2vec_embeddings(collection& col)
{
    auto embeddings = word2vec::builder{}
                      .vector_size(300)
                      .window_size(5)
                      .sample_threadhold(1e-5)
                      .num_negative_samples(5)
                      .num_threads(cstlm::num_cstlm_threads)
                      .num_iterations(5)
                      .min_freq_threshold(5)
                      .start_learning_rate(0.025)
                      .train_or_load(col);
    return embeddings;
}


rnnlm::LM load_or_create_rnnlm(collection& col, word2vec::embeddings& w2v_embeddings)
{
    auto rnn_lm = rnnlm::builder{}
                  .dropout(0.3)
                  .layers(2)
                  .vocab_threshold(nnlm::constants::VOCAB_THRESHOLD)
                  .hidden_dimensions(128)
                  .sampling(true)
                  .start_learning_rate(0.1)
                  .decay_rate(0.5)
                  .num_iterations(20)
                  .dev_file(col.file_map[KEY_DEV])
                  .train_or_load(col, w2v_embeddings);

    return rnn_lm;
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

    /* (3) create the cstlm model */
    auto rnnlm = load_or_create_rnnlm(col, word_embeddings);

    /* (4) parse test file */
    auto test_file      = col.file_map[KEY_TEST];
    auto test_sentences = load_and_parse_file(test_file, rnnlm);

    /* (5) evaluate sentences */
    evaluate_sentences(test_sentences, rnnlm);

    return 0;
}
