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
#include "common.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std::chrono;
using watch = std::chrono::high_resolution_clock;

namespace rnnlm {


// need this to load the w2v data into a lookup_parameter
struct ParameterInitEigenMatrix : public dynet::ParameterInit {
    ParameterInitEigenMatrix(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& m) : matrix(m) {}
    virtual void initialize_params(dynet::Tensor& values) const override
    {
        float* in = matrix.data();
#if HAVE_CUDA
        cstlm::LOG(cstlm::INFO) << "RNNLM CUDA INIT";
        cudaMemcpyAsync(values.v, in, sizeof(float) * matrix.size(), cudaMemcpyHostToDevice);
#else
        cstlm::LOG(cstlm::INFO) << "RNNLM EIGEN/CPU INIT";
        memcpy(values.v, in, sizeof(float) * matrix.size());
#endif
    }


private:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& matrix;
};

struct LM {
    dynet::Model                     model;
    dynet::LSTMBuilder               builder;
    dynet::LookupParameter           p_word_embeddings;
    dynet::Parameter                 p_R;
    dynet::Parameter                 p_bias;
    cstlm::vocab_uncompressed<false> full_vocab;
    cstlm::vocab_uncompressed<false> filtered_vocab;
    uint32_t                         layers;
    uint32_t                         w2v_vec_size;
    uint32_t                         hidden_dim;
    LM();

    LM(const LM& other)
        : model(other.model)
        , builder(other.builder)
        , p_word_embeddings(other.p_word_embeddings)
        , p_R(other.p_R)
        , p_bias(other.p_bias)
        , full_vocab(other.full_vocab)
        , filtered_vocab(other.filtered_vocab)
        , layers(other.layers)
        , w2v_vec_size(other.w2v_vec_size)
        , hidden_dim(other.hidden_dim)
    {
    }

    LM(LM&& other)
        : model(std::move(other.model))
        , builder(std::move(other.builder))
        , p_word_embeddings(std::move(other.p_word_embeddings))
        , p_R(std::move(other.p_R))
        , p_bias(std::move(other.p_bias))
        , full_vocab(std::move(other.full_vocab))
        , filtered_vocab(std::move(other.filtered_vocab))
        , layers(std::move(other.layers))
        , w2v_vec_size(std::move(other.w2v_vec_size))
        , hidden_dim(std::move(other.hidden_dim))
    {
    }

    LM& operator=(const LM& other)
    {
        if (this != &other) {
            LM tmp(other);          // re-use copy-constructor
            *this = std::move(tmp); // re-use move-assignment
        }
        return *this;
    }

    //! Assignment move operator
    LM& operator=(LM&& other)
    {
        if (this != &other) {
            model             = std::move(other.model);
            builder           = std::move(other.builder);
            p_word_embeddings = std::move(other.p_word_embeddings);
            p_R               = std::move(other.p_R);
            p_bias            = std::move(other.p_bias);
            full_vocab        = std::move(other.full_vocab);
            filtered_vocab    = std::move(other.filtered_vocab);
            layers            = std::move(other.layers);
            w2v_vec_size      = std::move(other.w2v_vec_size);
            hidden_dim        = std::move(other.hidden_dim);
        }
        return *this;
    }

    LM(std::string file_name) { load(file_name); }

    std::vector<std::vector<word_token>> parse_raw_sentences(std::string file_name) const
    {
        return sentence_parser::parse_from_raw(file_name, full_vocab, filtered_vocab);
    }

    void load(std::string file_name)
    {
        std::ifstream ifs(file_name);
        full_vocab.load(ifs);
        filtered_vocab.load(ifs);
        boost::archive::text_iarchive ia(ifs);
        ia >> model;
        ia >> layers;
        ia >> w2v_vec_size;
        ia >> hidden_dim;
        ia >> p_word_embeddings;
        ia >> p_R;
        ia >> p_bias;
        ia >> builder;
    }

    void store(std::string file_name)
    {
        std::ofstream ofs(file_name);
        full_vocab.serialize(ofs);
        filtered_vocab.serialize(ofs);
        boost::archive::text_oarchive oa(ofs);
        oa << model;
        oa << layers;
        oa << w2v_vec_size;
        oa << hidden_dim;
        oa << p_word_embeddings;
        oa << p_R;
        oa << p_bias;
        oa << builder;
    }

    LM(uint32_t                          _layers,
       uint32_t                          _hidden_dim,
       word2vec::embeddings&             emb,
       cstlm::vocab_uncompressed<false>& _full_vocab,
       cstlm::vocab_uncompressed<false>& _filtered_vocab)
        : builder(_layers, emb.cols(), _hidden_dim, model)
        , layers(_layers)
        , w2v_vec_size(emb.cols())
        , hidden_dim(_hidden_dim)
    {

        uint32_t vocab_size = _filtered_vocab.size();
        filtered_vocab      = _filtered_vocab;
        full_vocab          = _full_vocab;
        cstlm::LOG(cstlm::INFO) << "RNNLM W2V dimensions: " << vocab_size << "x" << w2v_vec_size;
        p_word_embeddings =
        model.add_lookup_parameters(vocab_size, {w2v_vec_size}, ParameterInitEigenMatrix(emb.data));
        p_R    = model.add_parameters({vocab_size, hidden_dim});
        p_bias = model.add_parameters({vocab_size});
    }


    dynet::expr::Expression build_lm_cgraph(const std::vector<word_token>& sentence,
                                            dynet::ComputationGraph&       cg,
                                            double                         m_dropout)
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
        for (size_t i = 0; i < sentence.size() - 1; i++) {
            auto i_x_t = dynet::expr::lookup(cg, p_word_embeddings, sentence[i].small_id);
            // y_t = RNN(x_t)
            auto i_y_t = builder.add_input(i_x_t);
            auto i_r_t = i_bias + i_R * i_y_t;

            // LogSoftmax followed by PickElement can be written in one step
            // using PickNegLogSoftmax
            auto i_err = dynet::expr::pickneglogsoftmax(i_r_t, sentence[i + 1].small_id);
            errs.push_back(i_err);
        }
        auto i_nerr = dynet::expr::sum(errs);
        return i_nerr;
    }

    sentence_eval evaluate_sentence_logprob(const std::vector<word_token>& sentence)
    {
        double                  logprob = 0.0;
        dynet::ComputationGraph cg;
        builder.new_graph(cg); // reset RNN builder for new graph
        builder.disable_dropout();

        builder.start_new_sequence();
        auto i_R    = dynet::expr::parameter(cg, p_R);    // hidden -> word rep parameter
        auto i_bias = dynet::expr::parameter(cg, p_bias); // word bias
        std::vector<dynet::expr::Expression> errs;
        for (size_t i = 0; i < sentence.size() - 1; i++) {
            auto i_x_t = dynet::expr::lookup(cg, p_word_embeddings, sentence[i].small_id);
            // y_t = RNN(x_t)
            auto i_y_t = builder.add_input(i_x_t);
            auto i_r_t = i_bias + i_R * i_y_t;

            // LogSoftmax followed by PickElement can be written in one step
            // using PickNegLogSoftmax
            if (sentence[i + 1].small_id != cstlm::UNKNOWN_SYM &&
                sentence[i + 1].big_id != cstlm::UNKNOWN_SYM) {
                auto i_err = dynet::expr::pickneglogsoftmax(i_r_t, sentence[i + 1].small_id);
                errs.push_back(i_err);
            }
        }
        auto loss_expr = dynet::expr::sum(errs);
        logprob        = as_scalar(cg.forward(loss_expr));
        return sentence_eval(logprob, errs.size());
    }
};

namespace constants {
}

namespace defaults {
const uint32_t LAYERS             = 2;
const float    DROPOUT            = 0.3f;
const uint32_t HIDDEN_DIM         = 512;
const bool     SAMPLE             = true;
const float    INIT_LEARNING_RATE = 0.1f;
const float    DECAY_RATE         = 0.5f;
const uint32_t VOCAB_THRESHOLD    = 30000;
const uint32_t NUM_ITERATIONS     = 5;
const uint32_t DECAY_AFTER_EPOCH  = 8;
}

struct builder {
    builder() {}

private:
    float       m_start_learning_rate = defaults::INIT_LEARNING_RATE;
    uint32_t    m_num_layers          = defaults::LAYERS;
    uint32_t    m_hidden_dim          = defaults::HIDDEN_DIM;
    bool        m_sampling            = defaults::SAMPLE;
    float       m_decay_rate          = defaults::DECAY_RATE;
    uint32_t    m_decay_after_epoch   = defaults::DECAY_AFTER_EPOCH;
    float       m_dropout             = defaults::DROPOUT;
    uint32_t    m_vocab_threshold     = defaults::VOCAB_THRESHOLD;
    uint32_t    m_num_iterations      = defaults::NUM_ITERATIONS;
    std::string m_dev_file            = "";

private:
    void output_params()
    {
        cstlm::LOG(cstlm::INFO) << "RNNLM layers: " << m_num_layers;
        cstlm::LOG(cstlm::INFO) << "RNNLM dropout: " << m_dropout;
        cstlm::LOG(cstlm::INFO) << "RNNLM hidden dimensions: " << m_hidden_dim;
        cstlm::LOG(cstlm::INFO) << "RNNLM sampling: " << m_sampling;
        cstlm::LOG(cstlm::INFO) << "RNNLM start learning rate: " << m_start_learning_rate;
        cstlm::LOG(cstlm::INFO) << "RNNLM decay rate: " << m_decay_rate;
        cstlm::LOG(cstlm::INFO) << "RNNLM decay after epoch: " << m_decay_after_epoch;
        cstlm::LOG(cstlm::INFO) << "RNNLM vocab threshold: " << m_vocab_threshold;
        cstlm::LOG(cstlm::INFO) << "RNNLM num iterations: " << m_num_iterations;
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

    builder& decay_after_epoch(uint32_t v)
    {
        m_decay_after_epoch = v;
        return *this;
    };

    builder& vocab_threshold(uint32_t v)
    {
        m_vocab_threshold = v;
        return *this;
    };

    builder& dev_file(std::string dev_file)
    {
        m_dev_file = dev_file;
        return *this;
    };

    std::string file_name(cstlm::collection& col)
    {
        std::stringstream fn;
        fn << col.path;
        fn << "/index/RNNLM-";
        fn << "l=" << m_num_layers << "-";
        fn << "d=" << m_dropout << "-";
        fn << "hd=" << m_hidden_dim << "-";
        fn << "s=" << m_sampling << "-";
        fn << "lr=" << m_start_learning_rate << "-";
        fn << "vt=" << m_vocab_threshold << "-";
        fn << "da=" << m_decay_after_epoch << "-";
        fn << "d=" << m_decay_rate;
        fn << ".dynet";
        return fn.str();
    }

    LM train_lm(cstlm::collection& col, word2vec::embeddings& w2v_embeddings,std::string out_file)
    {
        auto input_file = col.file_map[cstlm::KEY_SMALL_TEXT];

        cstlm::LOG(cstlm::INFO) << "RNNLM create/load full vocab";
        cstlm::vocab_uncompressed<false> vocab(col);

        cstlm::LOG(cstlm::INFO) << "RNNLM filter vocab";
        auto filtered_vocab = vocab.filter(input_file, m_vocab_threshold);

        cstlm::LOG(cstlm::INFO) << "RNNLM filter w2v embeddings";
        auto filtered_w2vemb = w2v_embeddings.filter(filtered_vocab);

        cstlm::LOG(cstlm::INFO) << "RNNLM parse sentences in training set";
        auto sentences = sentence_parser::parse(input_file, filtered_vocab);
        cstlm::LOG(cstlm::INFO) << "RNNLM sentences to process: " << sentences.size();

        cstlm::LOG(cstlm::INFO) << "RNNLM parse sentences in dev set";
        auto dev_sentences = sentence_parser::parse_from_raw(m_dev_file, vocab, filtered_vocab);
        cstlm::LOG(cstlm::INFO) << "RNNLM dev sentences to process: " << dev_sentences.size();

        // data will be stored here
        cstlm::LOG(cstlm::INFO) << "RNNLM init LM structure";
        LM rnnlm(m_num_layers, m_hidden_dim, filtered_w2vemb, vocab, filtered_vocab);

        cstlm::LOG(cstlm::INFO) << "RNNLM init SGD trainer";
        dynet::SimpleSGDTrainer sgd(rnnlm.model);
        sgd.eta0 = m_start_learning_rate;
        sgd.eta  = m_start_learning_rate;

        std::mt19937 gen(word2vec::consts::RAND_SEED);
        size_t       cur_sentence_id = 0;
        cstlm::LOG(cstlm::INFO) << "RNNLM start learning";

        double best_dev_pplx   = 999999.0;
	uint32_t finish_training = 0;
        for (size_t i = 1; i <= m_num_iterations; i++) {
            cstlm::LOG(cstlm::INFO) << "RNNLM shuffle sentences";
            std::shuffle(sentences.begin(), sentences.end(), gen);
            cur_sentence_id     = 0;
            size_t tokens       = 0;
            size_t total_tokens = 0;
            float  loss         = 0;
            for (const auto& sentence : sentences) {
                tokens += sentence.size(); // includes <S> and </S>
                total_tokens += sentence.size();
                dynet::ComputationGraph cg;

                auto loss_expr = rnnlm.build_lm_cgraph(sentence, cg, m_dropout);
                loss += dynet::as_scalar(cg.forward(loss_expr));

                cg.backward(loss_expr);
                sgd.update();

                if ((cur_sentence_id + 1) % ((sentences.size() / 20) + 1) == 0) {
                    // Print informations
                    cstlm::LOG(cstlm::INFO)
                    << "RNNLM [" << i << "/" << m_num_iterations << "] ("
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
            if (m_dev_file != "") {
                cstlm::LOG(cstlm::INFO) << "RNNLM evaluate dev pplx.";
                double log_probs = 0;
                size_t tokens    = 0;
                for (const auto& sentence : dev_sentences) {
                    auto eval_res = rnnlm.evaluate_sentence_logprob(sentence);
                    log_probs += eval_res.logprob;
                    tokens += eval_res.tokens;
                }
                double dev_pplx = exp(log_probs / tokens);
                cstlm::LOG(cstlm::INFO) << "RNNLM dev pplx= " << dev_pplx
                                        << " current best = " << best_dev_pplx;
                if (dev_pplx > best_dev_pplx) {
                    cstlm::LOG(cstlm::INFO) << "RNNLM dev pplx is getting worse. don't store.";
		    finish_training++;
                } else {
                    cstlm::LOG(cstlm::INFO) << "RNNLM dev pplx improved. we continue.";
                    best_dev_pplx = dev_pplx;
            	    cstlm::LOG(cstlm::INFO) << "RNNLM store to file.";
            	    rnnlm.store(out_file);
            	    cstlm::LOG(cstlm::INFO) << "RNNLM store to file done.";
		    finish_training = 0;
                }
            }

	    if( finish_training >= 3) {
		break;
	    } 

	    if(i >= m_decay_after_epoch) {
            	cstlm::LOG(cstlm::INFO) << "RNNLM update learning rate.";
            	sgd.eta *= m_decay_rate;
	    }
        }


        return rnnlm;
    }

    LM train_or_load(int                   argc,
                     char**                argv,
                     cstlm::collection&    col,
                     word2vec::embeddings& w2v_embeddings)
    {
        // (0) output params
        output_params();

        // (1) if exists. just load. otherwise construct
        auto rnnlm_file = file_name(col);
        if (!cstlm::utils::file_exists(rnnlm_file)) {
            dynet::initialize(argc, argv);
            auto rnn_lm = train_lm(col, w2v_embeddings,rnnlm_file);
        }
        cstlm::LOG(cstlm::INFO) << "RNNLM load from file.";
        dynet::cleanup();
        dynet::initialize(argc, argv);

        return LM(rnnlm_file);
    }
};
}
