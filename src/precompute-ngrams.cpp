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
    int         num_threads;
} cmdargs_t;

void print_usage(const char* program)
{
    fprintf(stdout, "%s -c -t\n", program);
    fprintf(stdout, "where\n");
    fprintf(stdout, "  -c <collection dir>  : the collection dir.\n");
    fprintf(stdout, "  -t <threads>         : limit the number of threads.\n");
}

cmdargs_t parse_args(int argc, const char* argv[])
{
    cmdargs_t args;
    int       op;
    args.collection_dir = "";
    args.num_threads    = 1;
    while ((op = getopt(argc, (char* const*)argv, "c:t:")) != -1) {
        switch (op) {
            case 'c':
                args.collection_dir = optarg;
                break;
            case 't':
                args.num_threads = atoi(optarg);
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

bool sortFunc(const std::vector<uint32_t>& p1, const std::vector<uint32_t>& p2)
{
    auto min_size = std::min(p1.size(), p2.size());
    for (size_t i = 0; i < min_size; i++) {
        if (p1[i] < p2[i]) return true;
        if (p1[i] > p2[i]) return false;
    }
    return false;
}

class uint32_vector_hasher {
public:
    std::size_t operator()(std::vector<uint32_t> const& vec) const
    {
        std::size_t seed = vec.size();
        for (auto& i : vec) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::vector<std::vector<uint32_t>>
create_sorted_ngrams(std::vector<std::vector<word_token>>& sentences, size_t ngram_size)
{
    std::unordered_set<std::vector<uint32_t>, uint32_vector_hasher> ngram_set;
    for (auto s : sentences) {
        auto itr = s.begin();
        auto end = itr + ngram_size;
        while (end <= s.end()) {
            std::vector<uint32_t> cur_gram;
            auto                  tmp = itr;
            while (tmp != end) {
                cur_gram.push_back(tmp->big_id);
                ++tmp;
            }
            ngram_set.insert(cur_gram);
            ++itr;
            ++end;
        }
    }
    std::vector<std::vector<uint32_t>> ngrams(ngram_set.begin(), ngram_set.end());
    std::sort(ngrams.begin(), ngrams.end(), sortFunc);
    return ngrams;
}

size_t compute_prefix_match(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b)
{
    auto min_size = std::min(a.size(), b.size());
    for (size_t i = 0; i < min_size; i++) {
        if (a[i] != b[i]) return i;
    }
    return min_size;
}


template <class t_cstlm>
std::unordered_map<uint64_t, std::vector<float>>
compute_ngrams(const std::vector<std::vector<uint32_t>>& ngrams,
               t_cstlm&                                  cstlm,
               cstlm::vocab_uncompressed<false>          f_vocab,
               size_t                                    id)
{
    std::unordered_map<uint64_t, std::vector<float>> cache;
    size_t            ngram_size = ngrams[0].size();
    static std::mutex m;

    using s_proxy_t = cstlm::LMQueryMKNE<t_cstlm>;
    s_proxy_t start(&cstlm, f_vocab, ngram_size, false, cache);

    std::stack<s_proxy_t> cmp_stack;
    cmp_stack.push(start);
    std::vector<uint32_t> prev_ngram;
    size_t                processed = 0;
    for (size_t i = 0; i < ngrams.size(); i++) {
        auto cur_ngram    = ngrams[i];
        auto prefix_match = compute_prefix_match(prev_ngram, cur_ngram);
        while (cmp_stack.size() > (prefix_match + 1)) {
            cmp_stack.pop();
        }
        for (size_t j = prefix_match; j < cur_ngram.size(); j++) {
            s_proxy_t copy = cmp_stack.top();
            copy.append_symbol(cur_ngram[j]);
            cmp_stack.push(copy);
        }
        prev_ngram = cur_ngram;
        if (processed == 100) {
            std::lock_guard<std::mutex> lock(m);
            cstlm::LOG(cstlm::INFO) << "[" << id << "] processed " << i + 1 << "/" << ngrams.size();
            cstlm::LOG(cstlm::INFO) << "[" << id << "] current cache size " << cache.size();
        }
        processed++;
    }

    return cache;
}

std::vector<std::vector<std::vector<uint32_t>>>
split_ngrams_chunks(const std::vector<std::vector<uint32_t>>& ngrams, size_t num_chunks)
{
    std::vector<std::vector<std::vector<uint32_t>>> chunks;

    size_t num_ngrams       = ngrams.size();
    size_t ngrams_per_chunk = num_ngrams / num_chunks;

    std::vector<std::vector<uint32_t>> chunk;
    chunk.push_back(ngrams[0]);
    for (size_t i = 1; i < num_ngrams; i++) {
        if (ngrams[i][0] != ngrams[i - 1][0]) {
            if (chunk.size() >= ngrams_per_chunk) {
                cstlm::LOG(cstlm::INFO) << "chunks[" << chunks.size() << "] size: " << chunk.size();
                chunks.push_back(chunk);
                chunk.clear();
            }
        }
        chunk.push_back(ngrams[i]);
    }
    if (chunk.size() != 0) {
        cstlm::LOG(cstlm::INFO) << "chunks[" << chunks.size() << "] size: " << chunk.size();
        chunks.push_back(chunk);
    }
    return chunks;
}

template <class t_cstlm>
void precompute_ngram_stats(
collection& col, t_cstlm& cstlm, size_t ngram_size, size_t vocab_size, int threads)
{
    auto input_file     = col.file_map[cstlm::KEY_SMALL_TEXT];
    auto dev_file       = col.file_map[cstlm::KEY_DEV];
    auto test_file      = col.file_map[cstlm::KEY_TEST];
    auto filtered_vocab = cstlm.vocab.filter(input_file, vocab_size);

    cstlm::LOG(cstlm::INFO) << "parse sentences in training set";
    auto sentences = sentence_parser::parse(input_file, filtered_vocab);
    cstlm::LOG(cstlm::INFO) << "sentences to process: " << sentences.size();

    cstlm::LOG(cstlm::INFO) << "parse sentences in dev set";
    auto dev_sents = sentence_parser::parse_from_raw(dev_file, cstlm.vocab, filtered_vocab);
    sentences.insert(sentences.end(), dev_sents.begin(), dev_sents.end());
    cstlm::LOG(cstlm::INFO) << "dev sentences to process: " << dev_sents.size();

    cstlm::LOG(cstlm::INFO) << "parse sentences in test set";
    auto test_sents = sentence_parser::parse_from_raw(test_file, cstlm.vocab, filtered_vocab);
    sentences.insert(sentences.end(), test_sents.begin(), test_sents.end());
    cstlm::LOG(cstlm::INFO) << "test sentences to process: " << test_file.size();

    auto ngrams = create_sorted_ngrams(sentences, ngram_size);
    cstlm::LOG(cstlm::INFO) << "ngrams to process: " << ngrams.size();

    size_t size_bytes = ngrams.size() * vocab_size * sizeof(float);
    float  size_mb    = float(size_bytes) / float(1024 * 1024);
    cstlm::LOG(cstlm::INFO) << "estimated size in mb: " << size_mb;

    cstlm::LOG(cstlm::INFO) << "splitting ngrams into " << threads << " chunks";
    auto split_ngrams = split_ngrams_chunks(ngrams, threads);
    cstlm::LOG(cstlm::INFO) << "generated chunks: " << split_ngrams.size();

    std::vector<std::future<std::unordered_map<uint64_t, std::vector<float>>>> future_results;
    for (size_t i = 0; i < split_ngrams.size(); i++) {
        const auto& chunk = split_ngrams[i];
        future_results.push_back(
        std::async(std::launch::async, [chunk, &cstlm, &filtered_vocab, i]() {
            return compute_ngrams(chunk, cstlm, filtered_vocab, i);
        }));
    }

    std::unordered_map<uint64_t, std::vector<float>> ngram_cache;
    for (auto& r : future_results) {
        const auto& cache = r.get();
        ngram_cache.insert(cache.begin(), cache.end());
        cstlm::LOG(cstlm::INFO) << "ngram_cache size: " << ngram_cache.size();
    }
}

template <class t_idx>
t_idx load_or_create_cstlm(collection& col)
{
    t_idx idx;
    auto  output_file =
    col.file_map[KEY_CSTLM_TEXT] + "-cstlm-" + sdsl::util::class_to_hash(idx) + ".sdsl";
    if (utils::file_exists(output_file)) {
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

int main(int argc, char** argv)
{
    enable_logging = true;

    /* parse command line */
    cmdargs_t args = parse_args(argc, (const char**)argv);

    /* (1) parse collection directory and create CSTLM index */
    collection col(args.collection_dir);
    col.file_map[KEY_CSTLM_TEXT] = col.file_map[KEY_BIG_TEXT];

    /* (2) create the cstlm model */
    auto           cstlm           = load_or_create_cstlm<wordlm>(col);
    const uint32_t ngram_size      = 5;
    const uint32_t vocab_threshold = nnlm::constants::VOCAB_THRESHOLD;

    /* (5) evaluate sentences */
    precompute_ngram_stats(col, cstlm, ngram_size, vocab_threshold, args.num_threads);

    return 0;
}
