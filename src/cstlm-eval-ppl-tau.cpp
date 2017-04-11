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
#include <unordered_set>


using namespace std;
using namespace dynet;
using namespace dynet::expr;
using namespace cstlm;

typedef struct cmdargs {
    std::string collection_dir;
    uint32_t    tau;
    bool        pplx;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c -t -T -x\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -t <threads>         : limit the number of threads.\n");
    fprintf(stdout, "  -T <tau>             : vocab threshold.\n");
    fprintf(stdout, "  -x                   : exclude unks in ppl eval.\n");
};

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int       op;
    args.collection_dir = "";
    args.tau            = 0;
    args.pplx           = false;
    while ((op = getopt(argc, (char* const*)argv, "c:t:T:")) != -1) {
        switch (op) {
            case 'c':
                args.collection_dir = optarg;
                break;
            case 't':
                num_cstlm_threads = std::atoi(optarg);
                break;
            case 'T':
                args.tau = std::atoi(optarg);
                break;
            case 'x':
                args.pplx = true;
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
                        size_t                                order,
                        cstlm::vocab_uncompressed<false>&     small_vocab,
                        uint64_t                              tau,
                        bool                                  eval_pplx)
{
    // (0) determine all rare symbols
    std::unordered_set<uint32_t> rare_syms;
    auto                         vitr = index.vocab.begin();
    auto                         vend = index.vocab.end();
    while (vitr != vend) {
        auto tok_str = vitr->first;
        try {
            small_vocab.token2id(tok_str);
        } catch (...) {
            /* could not convert -> word is rare */
            rare_syms.insert(vitr->second);
        }
        ++vitr;
    }

    double   logprobs  = 0;
    uint64_t M         = 0;
    uint64_t OOV       = 0;
    uint64_t sent_size = 0;
    for (const auto& sentence : sentences) {

        // (1) count the real OOVs
        for (size_t i = 0; i < sentence.size(); i++) {
            if (sentence[i].big_id == UNKNOWN_SYM) OOV++;
        }

        // (2) start the eval
        double                   final_score = 0;
        size_t                   num_tokens  = 0;
        cstlm::LMQueryMKN<t_idx> query(&index, order, false, false);
        bool                     first = true;
        for (const auto& word : sentence) {

            // (2c) if the real symbol is "rare" that is, not in the small vocab,
            //      we sum over all the rare symbols
            if (word.big_id != UNKNOWN_SYM) {
                if (rare_syms.count(word.big_id) != 0) { // rare word
                    // 1. sum over all rare words and add that
                    double rare_prob = 0.0;
                    for (const auto& rw : rare_syms) {
                        // we make a copy of the query object to iterate over all rare words
                        // and determine the probabilities they have in the given context
                        auto copy_obj = query;
                        auto score    = copy_obj.append_symbol(rw);
                        rare_prob += exp(score);
                    }
                    // 2. BUT we add the REAL word for conditioning
                    query.append_symbol(word.big_id);

                    if (first) {
                        first = false;
                        continue;
                    }

                    if (eval_pplx) {
                        // pplx we do not count UNKs!
                    } else {
                        num_tokens++;
                        final_score += rare_prob;
                    }


                } else { // not a rare word. just add the word
                    auto score = query.append_symbol(word.big_id);
                    if (first) {
                        first = false;
                        continue;
                    }
                    if (eval_pplx) {
                        // pplx we do not count UNKs!
                    } else {
                        num_tokens++;
                        final_score += score;
                    }
                }
            } else {
                auto score = query.append_symbol(word.big_id);
                // (2a) don't count <s> in prediction
                if (first) {
                    first = false;
                    continue;
                }
                if (eval_pplx) {
                    // pplx we do not count UNKs!
                } else {
                    num_tokens++;
                    final_score += score;
                }
            }
        }

        auto eval_res = sentence_logprob_kneser_ney2(index, sentence, M, order, true, false);
        logprobs += eval_res.logprob;
        M += eval_res.tokens;
        sent_size += sentence.size();
    }
    double perplexity = exp(-logprobs / M);
    LOG(INFO) << "CSTLM ORDER: " << order << " PPLX = " << std::setprecision(10) << perplexity
              << " (logprob = " << logprobs << ";predicted tokens = " << M << ";OOV = " << OOV
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
    col.file_map[KEY_CSTLM_TEXT] = col.file_map[KEY_BIG_TEXT];

    /* (2) create the cstlm model */
    auto cstlm = load_or_create_cstlm<wordlm>(col);

    auto filtered_vocab = cstlm.vocab.filter(col.file_map[KEY_BIG_TEXT], args.tau);

    /* (3) parse test file */
    auto test_sentences = load_and_parse_file(col, cstlm);

    /* (4) evaluate sentences */
    for (size_t i = 2; i < 20; i++) {
        evaluate_sentences(test_sentences, cstlm, i, filtered_vocab, args.tau, args.pplx);
    }

    return 0;
}
