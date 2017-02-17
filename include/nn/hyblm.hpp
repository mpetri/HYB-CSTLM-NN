#pragma once

#include "vocab_uncompressed.hpp"
#include "collection.hpp"
#include "logging.hpp"

#include <chrono>
#include <atomic>

#include "omp.h"

#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "word2vec.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "query_eigen.hpp"

using namespace std::chrono;
using watch = std::chrono::high_resolution_clock;

namespace HYBLM {


struct sentence_parser {

    static std::vector<std::vector<word_token>>
    parse(std::string                       file_name,
          cstlm::vocab_uncompressed<false>& filtered_vocab,
          cstlm::vocab_uncompressed<false>& vocab)
    {
        std::vector<std::vector<word_token>> sentences;
        sdsl::int_vector_buffer<0>           text(file_name);

        std::vector<word_token> cur;
        bool                    in_sentence = false;
        for (size_t i = 0; i < text.size(); i++) {
            auto sym = text[i];
            if (in_sentence == false && sym != cstlm::PAT_START_SYM) continue;

            if (sym == cstlm::PAT_START_SYM) {
                cur.push_back(sym);
                in_sentence = true;
                continue;
            }

            if (sym == cstlm::PAT_END_SYM) {
                cur.push_back(sym);
                if (cur.size() > 2) { // more than <s> and </s>?
                    sentences.push_back(cur);
                }
                cur.clear();
                in_sentence = false;
                continue;
            }

            // not start AND not END AND in sentence == true here
            // translate non-special ids to their small vocab id OR UNK
            auto small_vocab_id = vocab.big2small(sym);
            cur.emplace_back(small_vocab_id, sym);
        }
        return sentences;
    }
};

// need this to load the w2v data into a lookup_parameter
struct ParameterInitEigenMatrix : public dynet::ParameterInit {
    ParameterInitEigenMatrix(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& m) : matrix(m) {}
    virtual void initialize_params(dynet::Tensor& values) const override
    {
        float* in = matrix.data();
#if HAVE_CUDA
        cstlm::LOG(cstlm::INFO) << "HYBLM CUDA INIT";
        cudaMemcpyAsync(values.v, in, sizeof(float) * matrix.size(), cudaMemcpyHostToDevice);
#else
        cstlm::LOG(cstlm::INFO) << "HYBLM EIGEN/CPU INIT";
        memcpy(values.v, in, sizeof(float) * matrix.size());
#endif
    }


private:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& matrix;
};

template <class t_cstlm>
struct LM {
    dynet::Model                     model;
    dynet::LSTMBuilder               builder;
    dynet::LookupParameter           p_word_embeddings;
    dynet::Parameter                 p_R;
    dynet::Parameter                 p_bias;
    cstlm::vocab_uncompressed<false> filtered_vocab;
    uint32_t                         layers;
    uint32_t                         w2v_vec_size;
    uint32_t                         hidden_dim;
    uint32_t                         cstlm_ngramsize;
    t_cstlm*                         cstlm = nullptr;
    LM();

    LM(const t_cstlm& _cstlm, std::string file_name) : cstlm(&_cstlm) { load(file_name); }

    void load(std::string file_name)
    {
        std::ifstream ifs(file_name);
        vocab.load(ifs);
        boost::archive::text_iarchive ia(ifs);
        ia >> model;
        ia >> layers;
        ia >> w2v_vec_size;
        ia >> hidden_dim;
        ia >> p_word_embeddings;
        ia >> p_R;
        ia >> p_bias;
        ia >> builder;
        ia >> cstlm_ngramsize;
    }

    void store(std::string file_name)
    {
        std::ofstream ofs(file_name);
        vocab.serialize(ofs);
        boost::archive::text_oarchive oa(ofs);
        oa << model;
        oa << layers;
        oa << w2v_vec_size;
        oa << hidden_dim;
        oa << p_word_embeddings;
        oa << p_R;
        oa << p_bias;
        oa << builder;
        oa << cstlm_ngramsize;
    }

    LM(const t_cstlm& _cstlm, uint32_t _cstlm_ngramsize;
       uint32_t                          _layers,
       uint32_t                          _hidden_dim,
       word2vec::embeddings&             emb,
       cstlm::vocab_uncompressed<false>& _filtered_vocab)
        : builder(_layers, emb.cols(), _hidden_dim, model)
        , layers(_layers)
        , w2v_vec_size(emb.cols())
        , hidden_dim(_hidden_dim)
        , cstlm_ngramsize(_cstlm_ngramsize)
        , cstlm(&_cstlm)
    {

        uint32_t vocab_size = _filtered_vocab.size();
        filtered_vocab      = _filtered_vocab;
        cstlm::LOG(cstlm::INFO) << "HYBLM W2V dimensions: " << vocab_size << "x" << w2v_vec_size;
        p_word_embeddings =
        model.add_lookup_parameters(vocab_size, {w2v_vec_size}, ParameterInitEigenMatrix(emb.data));
        p_R    = model.add_parameters({vocab_size, hidden_dim});
        p_bias = model.add_parameters({vocab_size});
    }


    dynet::expr::Expression build_lm_cgraph(const std::vector<word_token> sentence,
                                            dynet::ComputationGraph&      cg,
                                            double                        m_dropout)
    {
        builder.new_graph(cg); // reset RNN builder for new graph
        if (m_dropout > 0) {
            builder.set_dropout(m_dropout);
        } else {
            builder.disable_dropout();
        }

        builder.start_new_sequence();
        auto i_R    = dynet::expr::parameter(cg, p_R);    // hidden -> word rep parameter
        auto i_bias = dynet::expr::parameter(cg, p_bias); // word bias
        std::vector<dynet::expr::Expression> errs;

        LMQueryMKNE<t_idx> cstlm_sentence(cstlm, filtered_vocab, cstlm_ngramsize, true);

        for (size_t i = 0; i < sentence.size() - 1; i++) {
            auto next_word_id = sentence[i].small_id;
            auto i_x_t        = dynet::expr::lookup(cg, p_word_embeddings, next_word_id);
            // y_t = RNN(x_t)
            auto i_y_t = builder.add_input(i_x_t);
            auto i_r_t = i_bias + i_R * i_y_t;

            // query cstlm
            auto next_word_bigid    = sentence[i].big_id;
            auto logprob_from_cstlm = cstlm_sentence.append_symbol(next_word_bigid);

            auto prod = logprob_from_cstlm * i_r_t;

            // LogSoftmax followed by PickElement can be written in one step
            // using PickNegLogSoftmax
            auto i_err = dynet::expr::pickneglogsoftmax(prod, sentence[i + 1]);
            errs.push_back(i_err);
        }
        auto i_nerr = dynet::expr::sum(errs);
        return i_nerr;
    }

    double evaluate_sentence_logprob(std::vector<word_token>& sentence)
    {
        double                  prob = 0.0;
        dynet::ComputationGraph cg;
        dynet::expr::Expression loss_expr = build_lm_cgraph(sentence, cg, 0.0);

        prob = as_scalar(cg.forward(loss_expr));
        return prob;
    }
};

namespace constants {
}

namespace defaults {
const uint32_t LAYERS                   = 2;
const float    DROPOUT                  = 0.3f;
const uint32_t HIDDEN_DIM               = 128;
const bool     SAMPLE                   = true;
const float    INIT_LEARNING_RATE       = 0.1f;
const float    DECAY_RATE               = 0.5f;
const uint32_t VOCAB_THRESHOLD          = 30000;
const uint32_t NUM_ITERATIONS           = 5;
const uint32_t DEFAULT_CSTLM_NGRAM_SIZE = 5;
}

struct builder {
    builder() {}

private:
    float    m_start_learning_rate = defaults::INIT_LEARNING_RATE;
    uint32_t m_num_layers          = defaults::LAYERS;
    uint32_t m_hidden_dim          = defaults::HIDDEN_DIM;
    bool     m_sampling            = defaults::SAMPLE;
    float    m_decay_rate          = defaults::DECAY_RATE;
    float    m_dropout             = defaults::DROPOUT;
    uint32_t m_vocab_threshold     = defaults::VOCAB_THRESHOLD;
    uint32_t m_num_iterations      = defaults::NUM_ITERATIONS;
    uint32_t m_cstlm_ngramsize     = defaults::DEFAULT_CSTLM_NGRAM_SIZE;

private:
    void output_params()
    {
        cstlm::LOG(cstlm::INFO) << "HYBLM layers: " << m_num_layers;
        cstlm::LOG(cstlm::INFO) << "HYBLM dropout: " << m_dropout;
        cstlm::LOG(cstlm::INFO) << "HYBLM hidden dimensions: " << m_hidden_dim;
        cstlm::LOG(cstlm::INFO) << "HYBLM sampling: " << m_sampling;
        cstlm::LOG(cstlm::INFO) << "HYBLM start learning rate: " << m_start_learning_rate;
        cstlm::LOG(cstlm::INFO) << "HYBLM decay rate: " << m_decay_rate;
        cstlm::LOG(cstlm::INFO) << "HYBLM vocab threshold: " << m_vocab_threshold;
        cstlm::LOG(cstlm::INFO) << "HYBLM num iterations: " << m_num_iterations;
        cstlm::LOG(cstlm::INFO) << "HYBLM cstlm ngramsize: " << m_cstlm_ngramsize;
    }


public:
    builder& dropout(float v)
    {
        m_dropout = v;
        return *this;
    };

    builder& hidden_dimensions(uint32_t hd)
    {
        m_hidden_dim = hd;
        return *this;
    };

    builder& sampling(bool v)
    {
        m_sampling = v;
        return *this;
    };

    builder& start_learning_rate(float alpha)
    {
        m_start_learning_rate = alpha;
        return *this;
    };

    builder& num_iterations(uint32_t n)
    {
        m_num_iterations = n;
        return *this;
    };

    builder& layers(uint32_t l)
    {
        m_num_layers = l;
        return *this;
    };

    builder& decay_rate(float v)
    {
        m_decay_rate = v;
        return *this;
    };

    builder& vocab_threshold(uint32_t v)
    {
        m_vocab_threshold = v;
        return *this;
    };

    builder& cstlm_ngramsize(uint32_t v)
    {
        m_cstlm_ngramsize = v;
        return *this;
    };

    std::string file_name(cstlm::collection& col)
    {
        std::stringstream fn;
        fn << col.path;
        fn << "/index/HYBLM-";
        fn << "l=" << m_num_layers << "-";
        fn << "d=" << m_dropout << "-";
        fn << "hd=" << m_hidden_dim << "-";
        fn << "s=" << m_sampling << "-";
        fn << "n=" << m_cstlm_ngramsize << "-";
        fn << "lr=" << m_start_learning_rate << "-";
        fn << "vt=" << m_vocab_threshold << "-";
        fn << "decay=" << m_decay_rate;
        fn << ".dynet";
        return fn.str();
    }

    template <class t_cstlm>
    LM<t_cstlm>
    train_lm(cstlm::collection& col, const t_cstlm& cstlm, word2vec::embeddings& w2v_embeddings)
    {
        auto input_file = col.file_map[cstlm::KEY_SMALL_TEXT];

        cstlm::LOG(cstlm::INFO) << "HYBLM filter vocab";
        auto filtered_vocab = cstlm.vocab.filter(input_file, m_vocab_threshold);

        cstlm::LOG(cstlm::INFO) << "HYBLM filter w2v embeddings";
        auto filtered_w2vemb = w2v_embeddings.filter(filtered_vocab);

        cstlm::LOG(cstlm::INFO) << "HYBLM parse sentences";
        auto sentences = sentence_parser::parse(input_file, filtered_vocab, cstlm.vocab);
        cstlm::LOG(cstlm::INFO) << "HYBLM sentences to process: " << sentences.size();

        // data will be stored here
        cstlm::LOG(cstlm::INFO) << "HYBLM init LM structure";
        LM hyblm(
        cstlm, m_cstlm_ngramsize, m_num_layers, m_hidden_dim, filtered_w2vemb, filtered_vocab);

        cstlm::LOG(cstlm::INFO) << "HYBLM init SGD trainer";
        dynet::SimpleSGDTrainer sgd(HYBLM.model);
        sgd.eta0 = m_start_learning_rate;


        std::mt19937 gen(word2vec::consts::RAND_SEED);
        size_t       cur_sentence_id = 0;
        cstlm::LOG(cstlm::INFO) << "HYBLM start learning";

        for (size_t i = 0; i < m_num_iterations; i++) {
            cstlm::LOG(cstlm::INFO) << "HYBLM shuffle sentences";
            std::shuffle(sentences.begin(), sentences.end(), gen);
            cur_sentence_id     = 0;
            size_t tokens       = 0;
            size_t total_tokens = 0;
            float  loss         = 0;
            for (const auto& sentence : sentences) {
                tokens += sentence.size(); // includes <S> and </S>
                total_tokens += sentence.size();
                dynet::ComputationGraph cg;

                auto loss_expr = hyblm.build_lm_cgraph(sentence, cg, m_dropout);
                loss += dynet::as_scalar(cg.forward(loss_expr));

                cg.backward(loss_expr);
                sgd.update();

                if ((cur_sentence_id + 1) % ((sentences.size() / 20) + 1) == 0) {
                    // Print informations
                    cstlm::LOG(cstlm::INFO)
                    << "HYBLM [" << i + 1 << "/" << m_num_iterations << "] ("
                    << 100 * float(cur_sentence_id) / float(sentences.size() + 1)
                    << "%) S = " << cur_sentence_id << " T = " << total_tokens
                    << " eta = " << sgd.eta << " E = " << (loss / (tokens + 1))
                    << " ppl=" << exp(loss / (tokens + 1)) << ' ';
                    // Reinitialize loss
                    loss   = 0;
                    tokens = 0;
                }
                cur_sentence_id++;
            }
            cstlm::LOG(cstlm::INFO) << "HYBLM update learning rate.";
            sgd.eta *= m_decay_rate;
        }


        return HYBLM;
    }

    template <class t_cstlm>
    LM train_or_load(cstlm::collection&    col,
                     const t_cstlm&        cstlm,
                     word2vec::embeddings& w2v_embeddings)
    {
        // (0) output params
        output_params();

        // (1) if exists. just load
        auto HYBLM_file = file_name(col);
        if (cstlm::utils::file_exists(HYBLM_file)) {
            return LM(cstlm, HYBLM_file);
        }

        // (3) train
        auto hyb_lm = train_lm(col, cstlm, w2v_embeddings);

        hyb_lm.store(HYBLM_file);

        return hyb_lm;
    }
};
}
