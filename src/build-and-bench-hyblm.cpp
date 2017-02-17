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
    int       op;
    args.collection_dir = "";
    args.test_file      = "";
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
    std::ifstream                      ifile(file_name);
    LOG(INFO) << "reading input file '" << file_name << "'";
    std::string line;
    while (std::getline(ifile, line)) {
        auto                  line_tokens = utils::parse_line(line, false);
        std::vector<uint32_t> tokens;
        tokens.push_back(PAT_START_SYM);
        for (const auto& token : line_tokens) {
            auto num = index.vocab.token2id(token, UNKNOWN_SYM);
            tokens.push_back(num);
        }
        tokens.push_back(PAT_END_SYM);
        LOG(INFO) << "S(" << sentences.size() << ") = " << sentence_to_str(tokens, index);
        sentences.push_back(tokens);
    }
    LOG(INFO) << "found " << sentences.size() << " sentences";
    return sentences;
}

void evaluate_sentences(std::vector<std::vector<uint32_t>>& sentences, rnnlm::LM& rnn_lm)
{
    double perplexity          = 0;
    double num_words_predicted = 0;
    for (auto sentence : sentences) {
        double sentenceprob = rnn_lm.evaluate_sentence_logprob(sentence);
        num_words_predicted += (sentence.size() - 1);
        perplexity += sentenceprob;
    }
    perplexity = perplexity / num_words_predicted;
    LOG(INFO) << "RNNLM PPLX = " << std::setprecision(10) << pow(10, -perplexity);
    ;
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


template <class t_idx>
t_idx load_or_create_cstlm(collection& col)
{
    t_idx idx;
    auto  output_file = col.path + "/index/index-" + col.file_map[KEY_CSTLM_TEXT] + "-" +
                       sdsl::util::class_to_hash(idx) + ".sdsl";
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

template <class t_cstlm>
rnnlm::LM
load_or_create_rnnlm(collection& col, const t_cstlm& cstlm, word2vec::embeddings& w2v_embeddings)
{
    auto rnn_lm = rnnlm::builder{}
                  .dropout(0.3)
                  .layers(2)
                  .vocab_threshold(30000)
                  .hidden_dimensions(128)
                  .sampling(true)
                  .start_learning_rate(0.1)
                  .decay_rate(0.5)
                  .num_iterations(5)
                  .train_or_load(col, cstlm, w2v_embeddings);

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
    col.file_map[KEY_CSTLM_TEXT] = col.file_map[KEY_BIG_TEXT];

    /* (2) create the cstlm model */
    auto cstlm = load_or_create_cstlm<wordlm>(col, true);

    /* (2) load the word2vec embeddings */
    auto word_embeddings = load_or_create_word2vec_embeddings(col);

    /* (3) create the cstlm model */
    auto hyblm = load_or_create_hyblm(col, cstlm, word_embeddings);

    /* (4) parse test file */
    auto test_sentences = load_and_parse_file(args.test_file, hyblm);

    /* (5) evaluate sentences */
    evaluate_sentences(test_sentences, hyblm);

    return 0;
}