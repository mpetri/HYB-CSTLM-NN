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

using namespace std::chrono;
using watch = std::chrono::high_resolution_clock;

namespace rnnlm {

struct sentence_parser {

	static std::vector<std::vector<uint32_t>> parse(std::string						  file_name,
													cstlm::vocab_uncompressed<false>& vocab)
	{
		std::vector<std::vector<uint32_t>> sentences;
		sdsl::int_vector_buffer<0>		   text(file_name);

		std::vector<uint32_t> cur;
		bool				  in_sentence = false;
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
			cur.push_back(small_vocab_id);
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
		cudaMemcpyAsync(values.v, in, sizeof(float) * matrix.size(), cudaMemcpyHostToDevice);
#else
		memcpy(values.v, in, sizeof(float) * matrix.size());
#endif
	}

private:
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& matrix;
};

struct LM {
	dynet::LSTMBuilder				 builder;
	dynet::LookupParameter			 p_word_embeddings;
	dynet::Parameter				 p_R;
	dynet::Parameter				 p_bias;
	cstlm::vocab_uncompressed<false> vocab;
	dynet::Model					 model;
	uint32_t						 layers;
	uint32_t						 w2v_vec_size;
	uint32_t						 hidden_dim;
	LM();

	LM(std::string file_name) { load(file_name); }

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
	}

	LM(uint32_t							 _layers,
	   uint32_t							 _hidden_dim,
	   word2vec::embeddings&			 emb,
	   cstlm::vocab_uncompressed<false>& filtered_vocab)
		: builder(_layers, emb.cols(), _hidden_dim, model)
		, layers(_layers)
		, w2v_vec_size(emb.cols())
		, hidden_dim(_hidden_dim)
	{

		uint32_t vocab_size = filtered_vocab.size();
		vocab				= filtered_vocab;
		cstlm::LOG(cstlm::INFO) << "RNNLM W2V dimensions: " << vocab_size << "x" << w2v_vec_size;
		p_word_embeddings =
		model.add_lookup_parameters(vocab_size, {w2v_vec_size}, ParameterInitEigenMatrix(emb.data));
		p_R	= model.add_parameters({vocab_size, hidden_dim});
		p_bias = model.add_parameters({vocab_size});
	}


	dynet::expr::Expression build_lm_cgraph(const std::vector<uint32_t> sentence,
											dynet::ComputationGraph&	cg,
											double						m_dropout)
	{
		builder.new_graph(cg); // reset RNN builder for new graph
		if (m_dropout > 0) {
			builder.set_dropout(m_dropout);
		} else {
			builder.disable_dropout();
		}

		builder.start_new_sequence();
		auto i_R	= dynet::expr::parameter(cg, p_R);	// hidden -> word rep parameter
		auto i_bias = dynet::expr::parameter(cg, p_bias); // word bias
		std::vector<dynet::expr::Expression> errs;
		for (size_t i = 0; i < sentence.size() - 1; i++) {
			auto i_x_t = dynet::expr::lookup(cg, p_word_embeddings, sentence[i]);
			// y_t = RNN(x_t)
			auto i_y_t = builder.add_input(i_x_t);
			auto i_r_t = i_bias + i_R * i_y_t;

			// LogSoftmax followed by PickElement can be written in one step
			// using PickNegLogSoftmax
			auto i_err = dynet::expr::pickneglogsoftmax(i_r_t, sentence[i + 1]);
			errs.push_back(i_err);
		}
		auto i_nerr = dynet::expr::sum(errs);
		return i_nerr;
	}

	double evaluate_sentence_logprob(std::vector<uint32_t>& sentence)
	{
		double					prob = 0.0;
		dynet::ComputationGraph cg;
		dynet::expr::Expression loss_expr = build_lm_cgraph(sentence, cg, 0.0);

		prob = as_scalar(cg.forward(loss_expr));
		return prob;
	}
};

namespace constants {
}

namespace defaults {
const uint32_t LAYERS			  = 2;
const float	DROPOUT			  = 0.3f;
const uint32_t HIDDEN_DIM		  = 128;
const bool	 SAMPLE			  = true;
const float	INIT_LEARNING_RATE = 0.1f;
const float	DECAY_RATE		  = 0.5f;
const uint32_t VOCAB_THRESHOLD	= 30000;
const uint32_t NUM_ITERATIONS	 = 5;
}

struct builder {
	builder() {}

private:
	float	m_start_learning_rate = defaults::INIT_LEARNING_RATE;
	uint32_t m_num_layers		   = defaults::LAYERS;
	uint32_t m_hidden_dim		   = defaults::HIDDEN_DIM;
	bool	 m_sampling			   = defaults::SAMPLE;
	float	m_decay_rate		   = defaults::DECAY_RATE;
	float	m_dropout			   = defaults::DROPOUT;
	uint32_t m_vocab_threshold	 = defaults::VOCAB_THRESHOLD;
	uint32_t m_num_iterations	  = defaults::NUM_ITERATIONS;

private:
	void output_params()
	{
		cstlm::LOG(cstlm::INFO) << "RNNLM layers: " << m_num_layers;
		cstlm::LOG(cstlm::INFO) << "RNNLM dropout: " << m_dropout;
		cstlm::LOG(cstlm::INFO) << "RNNLM hidden dimensions: " << m_hidden_dim;
		cstlm::LOG(cstlm::INFO) << "RNNLM sampling: " << m_sampling;
		cstlm::LOG(cstlm::INFO) << "RNNLM start learning rate: " << m_start_learning_rate;
		cstlm::LOG(cstlm::INFO) << "RNNLM decay rate: " << m_decay_rate;
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

	builder& vocab_threshold(uint32_t v)
	{
		m_vocab_threshold = v;
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
		fn << "decay=" << m_decay_rate;
		fn << ".dynet";
		return fn.str();
	}

	LM train_lm(cstlm::collection& col, word2vec::embeddings& w2v_embeddings)
	{
		auto input_file = col.file_map[cstlm::KEY_SMALL_TEXT];

		cstlm::LOG(cstlm::INFO) << "RNNLM create/load full vocab";
		cstlm::vocab_uncompressed<false> vocab(col);

		cstlm::LOG(cstlm::INFO) << "RNNLM filter vocab";
		auto filtered_vocab = vocab.filter(input_file, m_vocab_threshold);

		cstlm::LOG(cstlm::INFO) << "RNNLM filter w2v embeddings";
		auto filtered_w2vemb = w2v_embeddings.filter(filtered_vocab);

		cstlm::LOG(cstlm::INFO) << "RNNLM parse sentences";
		auto sentences = sentence_parser::parse(input_file, filtered_vocab);
		cstlm::LOG(cstlm::INFO) << "RNNLM sentences to process: " << sentences.size();

		// data will be stored here
		LM rnnlm(m_num_layers, m_hidden_dim, filtered_w2vemb, filtered_vocab);

		dynet::SimpleSGDTrainer sgd(rnnlm.model);
		sgd.eta0 = m_start_learning_rate;


		std::mt19937 gen(word2vec::consts::RAND_SEED);
		shuffle(sentences.begin(), sentences.end(), gen);
		size_t cur_sentence_id = 0;


		cstlm::LOG(cstlm::INFO) << "RNNLM start learning";

		for (size_t i = 0; i < m_num_iterations; i++) {
			std::shuffle(sentences.begin(), sentences.end(), gen);
			cur_sentence_id		= 0;
			size_t tokens		= 0;
			size_t total_tokens = 0;
			float  loss			= 0;
			for (const auto& sentence : sentences) {
				tokens += sentence.size(); // includes <S> and </S>
				total_tokens += sentence.size();
				dynet::ComputationGraph cg;

				auto loss_expr = rnnlm.build_lm_cgraph(sentence, cg, m_dropout);
				loss += dynet::as_scalar(cg.forward(loss_expr));

				cg.backward(loss_expr);
				sgd.update();

				if ((cur_sentence_id + 1) % ((sentences.size() / 100000) + 1) == 0) {
					// Print informations
					cstlm::LOG(cstlm::INFO)
					<< "RNNLM [" << i + 1 << "/" << m_num_iterations << "] ("
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
			cstlm::LOG(cstlm::INFO) << "RNNLM update learning rate.";
			sgd.eta *= m_decay_rate;
		}


		return rnnlm;
	}

	LM train_or_load(cstlm::collection& col, word2vec::embeddings& w2v_embeddings)
	{
		// (0) output params
		output_params();

		// (1) if exists. just load
		auto rnnlm_file = file_name(col);
		if (cstlm::utils::file_exists(rnnlm_file)) {
			return LM(rnnlm_file);
		}

		// (3) train
		auto rnn_lm = train_lm(col, w2v_embeddings);

		rnn_lm.store(rnnlm_file);

		return rnn_lm;
	}
};
}
