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

using namespace std::chrono;
using watch = std::chrono::high_resolution_clock;

namespace hyblm {

struct sentence_parser {

	static std::vector<std::vector<uint32_t>> parse(std::string file_name)
	{
		std::vector<std::vector<uint32_t>> sentences;
		sdsl::int_vector_buffer<0>		   text(file_name);

		std::vector<uint32_t> cur;
		bool				  in_sentence = false;
		for (size_t i = 0; i < text.size(); i++) {
			auto sym = text.size();
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
			cur.push_back(sym);
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
	dynet::LSTMBuilder	 builder;
	dynet::LookupParameter p_word_embeddings;
	dynet::Parameter	   p_R;
	dynet::Parameter	   p_bias;

	LM(std::string file_name) { load(file_name); }

	void load(std::string file_name) { std::cout << "TODO: load hyblm from file " << file_name; }

	void store(std::string file_name) { std::cout << "TODO: store hyblm to file " << file_name; }

	LM(dynet::Model& model, uint32_t layers, uint32_t hidden_dim, word2vec::embeddings& emb)
		: builder(layers, emb.rows(), emb.cols(), model)
	{
		p_word_embeddings =
		model.add_lookup_parameters(emb.rows(), {emb.cols()}, ParameterInitEigenMatrix(emb.data));
		p_R	= model.add_parameters({emb.rows(), hidden_dim});
		p_bias = model.add_parameters({emb.rows()});
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

private:
	void output_params()
	{
		cstlm::LOG(cstlm::INFO) << "HYBLM layers: " << m_num_layers;
		cstlm::LOG(cstlm::INFO) << "HYBLM dropout: " << m_dropout;
		cstlm::LOG(cstlm::INFO) << "HYBLM hidden dimensions: " << m_hidden_dim;
		cstlm::LOG(cstlm::INFO) << "HYBLM sampling: " << m_sampling;
		cstlm::LOG(cstlm::INFO) << "HYBLM start learning rate: " << m_start_learning_rate;
		cstlm::LOG(cstlm::INFO) << "HYBLM decay rate: " << m_decay_rate;
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
		fn << "decay=" << m_decay_rate;
		fn << ".dynet";
		return fn.str();
	}

	dynet::expr::Expression
	build_lm_cgraph(const std::vector<uint32_t> sentence, LM& hyblm, dynet::ComputationGraph& cg)
	{
		hyblm.builder.new_graph(cg); // reset RNN builder for new graph
		if (m_dropout > 0) {
			hyblm.builder.set_dropout(m_dropout);
		} else {
			hyblm.builder.disable_dropout();
		}

		hyblm.builder.start_new_sequence();
		auto i_R	= dynet::expr::parameter(cg, hyblm.p_R);	// hidden -> word rep parameter
		auto i_bias = dynet::expr::parameter(cg, hyblm.p_bias); // word bias
		std::vector<dynet::expr::Expression> errs;
		for (size_t i = 0; i < sentence.size() - 1; i++) {
			auto i_x_t = dynet::expr::lookup(cg, hyblm.p_word_embeddings, sentence[i]);
			// y_t = RNN(x_t)
			auto i_y_t = hyblm.builder.add_input(i_x_t);
			auto i_r_t = i_bias + i_R * i_y_t;

			// LogSoftmax followed by PickElement can be written in one step
			// using PickNegLogSoftmax
			auto i_err = dynet::expr::pickneglogsoftmax(i_r_t, sentence[i + 1]);
			errs.push_back(i_err);
		}
		auto i_nerr = dynet::expr::sum(errs);
		return i_nerr;
	}

	template <class t_cstlmidx>
	LM train_lm(cstlm::vocab_uncompressed<false>& vocab,
				word2vec::embeddings&			  w2v_emb,
				t_cstlmidx&						  cstlm,
				cstlm::collection&				  col)
	{
		auto sentences = sentence_parser::parse(col.file_map[cstlm::KEY_SMALLTEXT]);

		dynet::Model			model;
		dynet::SimpleSGDTrainer sgd(model);
		sgd.eta0 = m_start_learning_rate;

		// data will be stored here
		LM hyblm(model, m_num_layers, m_hidden_dim, w2v_emb);

		// TODO do we shuffle the sentences first?
		for (const auto& sentence : sentences) {
			dynet::ComputationGraph cg;

			auto loss_expr = build_lm_cgraph(sentence, hyblm, cg);
			auto loss	  = dynet::as_scalar(cg.forward(loss_expr));
			std::cout << "loss = " << loss << std::endl;
			cg.backward(loss_expr);
			sgd.update();
		}


		return hyblm;
	}

	template <class t_cstlmidx>
	LM train_or_load(cstlm::collection& col, t_cstlmidx& cstlm, word2vec::embeddings& emb)
	{
		// (0) output params
		output_params();

		// (1) if exists. just load
		auto hyblm_file = file_name(col);
		if (cstlm::utils::file_exists(hyblm_file)) {
			return LM(hyblm_file);
		}

		// (2) load vocab
		cstlm::LOG(cstlm::INFO) << "create vocab";
		cstlm::vocab_uncompressed<false> vocab(col);

		// (3) train
		auto hyb_lm = train_lm(vocab, emb, cstlm, col);

		hyb_lm.store(hyblm_file);

		return hyb_lm;
	}
};
}
