#pragma once

#include "vocab_uncompressed.hpp"
#include "collection.hpp"
#include "logging.hpp"

#include <chrono>

using namespace std::chrono;
using watch = std::chrono::high_resolution_clock;

namespace word2vec {

namespace consts {
const uint64_t RAND_SEED		  = 0XBEEF;
const uint64_t EXP_TABLE_SIZE	 = 1000;
const uint64_t MAX_EXP			  = 6;
const uint64_t MAX_SENTENCE_LEN   = 1000;
const uint64_t UNIGRAM_TABLE_SIZE = 1e8;
}

namespace constants {

const uint32_t DEFAULT_VEC_SIZE					= 100;
const float	DEFAULT_LEARNING_RATE			= 0.025f;
const uint32_t DEFAULT_WINDOW_SIZE				= 5;
const float	DEFAULT_SAMPLE_THRESHOLD			= 1e-3;
const uint32_t DEFAULT_NUM_NEG_SAMPLES			= 5;
const uint32_t DEFAULT_NUM_ITERATIONS			= 5;
const uint32_t DEFAULT_MIN_FREQ_THRES			= 5;
const uint32_t DEFAULT_BATCH_SIZE				= 2 * DEFAULT_WINDOW_SIZE + 1;
const bool	 DEFAULT_USE_HIERARCHICAL_SOFTMAX = false;
const bool	 DEFAULT_USE_CBOW					= false;
}

struct embeddings {
	embeddings() {}
	embeddings(std::string file_name) {}

	void store(std::string file_name) {}
};

struct net_state {
	std::vector<float> syn0;
	std::vector<float> syn1;
	std::vector<float> syn1neg;
};


template <class t_itr>
struct sentence_parser {
private:
	t_itr							  beg;
	t_itr							  end;
	t_itr							  cur;
	cstlm::vocab_uncompressed<false>& vocab;
	float							  sample_threshold;
	float							  min_freq;
	std::array<uint32_t, consts::MAX_SENTENCE_LEN> m_sentence_buf;
	std::mt19937					 gen			  = std::mt19937(consts::RAND_SEED);
	std::uniform_real_distribution<> subsampling_dist = std::uniform_real_distribution<>(0, 1);
	uint64_t						 m_cur_sentence_len;

public:
	const std::array<uint32_t, consts::MAX_SENTENCE_LEN>& cur_sentence = m_sentence_buf;

public:
	sentence_parser(
	t_itr _beg, t_itr _end, cstlm::vocab_uncompressed<false>& v, float thres, float mf)
		: beg(_beg), end(_end), cur(beg), vocab(v), sample_threshold(thres), min_freq(mf)
	{
	}

	uint64_t cur_size() { return m_cur_sentence_len; }

	uint64_t cur_offset_in_text() { return std::distance(beg, cur); }

	bool next_sentence()
	{
		size_t offset = 0;
		bool   add	= false;
		while (cur != end) {
			auto sym = *cur;
			if (add) {

				double freq = vocab.freq[sym];

				// drop words below min-freq
				if (freq < min_freq) continue;

				// The subsampling randomly discards frequent words while keeping the ranking same
				if (sample_threshold > 0) {
					double cprob = freq / double(vocab.total_freq);
					auto   prob  = 1.0 - sqrt(sample_threshold / cprob);
					auto   gprob = subsampling_dist(gen);
					if (prob < gprob) continue;
				}

				m_sentence_buf[offset++] = sym;
			}
			if (sym == cstlm::PAT_START_SYM) { // sentence start sym found
				add = true;
			}
			if (sym == cstlm::PAT_END_SYM) { // sentence start sym found
				break;
			}
			if (offset == consts::MAX_SENTENCE_LEN) {
				break;
			}
			++cur;
		}
		m_cur_sentence_len = offset;
		if (cur == end && offset == 0) {
			return false;
		}
		return true;
	}
};

struct builder {
	builder()
	{
		for (size_t i = 0; i < consts::EXP_TABLE_SIZE; i++) {
			m_expTable[i] = exp((i / (float)consts::EXP_TABLE_SIZE * 2 - 1) *
								consts::MAX_EXP);				 // Precompute the exp() table
			m_expTable[i] = m_expTable[i] / (m_expTable[i] + 1); // Precompute f(x) = x / (x + 1)
		}
	}

private:
	float	m_start_learning_rate  = constants::DEFAULT_LEARNING_RATE;
	uint32_t m_vector_size			= constants::DEFAULT_VEC_SIZE;
	uint32_t m_window_size			= constants::DEFAULT_WINDOW_SIZE;
	float	m_sample_threshold		= constants::DEFAULT_SAMPLE_THRESHOLD;
	uint32_t m_num_negative_samples = constants::DEFAULT_NUM_NEG_SAMPLES;
	uint32_t m_num_iterations		= constants::DEFAULT_NUM_ITERATIONS;
	uint32_t m_min_freq_threshold   = constants::DEFAULT_MIN_FREQ_THRES;
	uint32_t m_batch_size			= constants::DEFAULT_BATCH_SIZE;
	uint32_t m_num_threads			= std::thread::hardware_concurrency();
	bool	 m_use_cbow				= constants::DEFAULT_USE_CBOW;
	bool	 m_hierarchical_softmax = constants::DEFAULT_USE_HIERARCHICAL_SOFTMAX;

private:
	std::vector<uint32_t> m_unigram_neg_table;
	net_state			  m_net_state;
	std::array<float, consts::EXP_TABLE_SIZE + 1> m_expTable;

private:
	void output_params()
	{
		std::cout << "number of threads: " << m_num_threads << std::endl;
		std::cout << "number of iterations: " << m_num_iterations << std::endl;
		std::cout << "hidden size: " << m_vector_size << std::endl;
		std::cout << "number of negative samples: " << m_num_negative_samples << std::endl;
		std::cout << "use hierchical softmax: " << m_hierarchical_softmax << std::endl;
		std::cout << "window size: " << m_window_size << std::endl;
		std::cout << "batch size: " << m_num_threads << std::endl;
		std::cout << "min freq: " << m_min_freq_threshold << std::endl;
		std::cout << "starting learning rate: " << m_start_learning_rate << std::endl;
		if (m_use_cbow) {
			std::cout << "model: continuous back of words (CBOW)" << std::endl;
		} else {
			std::cout << "model: skip-gram (SG)" << std::endl;
		}
	}


	void init_net_state(cstlm::vocab_uncompressed<false>& vocab)
	{
		m_net_state.syn0.resize(vocab.size() * m_vector_size);

		// random initialization of first layer
		// from other implementations: rand in [-0.5,0.5]/vector_size
		std::mt19937					 gen(consts::RAND_SEED);
		float							 min_rand = -0.5f / float(m_vector_size);
		float							 max_rand = 0.5f / float(m_vector_size);
		std::uniform_real_distribution<> dist(min_rand, max_rand);
		for (size_t i = 0; i < m_net_state.syn0.size(); i++) {
			m_net_state.syn0[i] = dist(gen);
		}

		if (m_hierarchical_softmax == true) {
			m_net_state.syn1.resize(vocab.size() * m_vector_size);
			for (size_t i = 0; i < m_net_state.syn1.size(); i++) {
				m_net_state.syn1[i] = 0;
			}
		}

		if (m_num_negative_samples > 0) {
			m_net_state.syn1neg.resize(vocab.size() * m_vector_size);
			for (size_t i = 0; i < m_net_state.syn1neg.size(); i++) {
				m_net_state.syn1neg[i] = 0;
			}
		}
	}

	void init_unigram_table(cstlm::vocab_uncompressed<false>& vocab)
	{
		m_unigram_neg_table.resize(consts::UNIGRAM_TABLE_SIZE);

		// from the paper -> power 3/4
		const float power			= 0.75f;
		double		train_words_pow = 0.0;
		for (size_t i = 0; i < vocab.size(); i++) {
			train_words_pow += pow(vocab.freq[i], power);
		}

		uint32_t i			= 0;
		float	d1			= pow(vocab.freq[i], power) / train_words_pow;
		size_t   table_size = m_unigram_neg_table.size();
		for (size_t a = 0; a < m_unigram_neg_table.size(); a++) {
			m_unigram_neg_table[a] = i;
			if (float(a) / float(table_size) > d1) {
				i++;
				d1 += pow(vocab.freq[i], power) / train_words_pow;
			}
			if (i >= vocab.size()) i = vocab.size() - 1;
		}
	}

	void create_huffman_tree(cstlm::vocab_uncompressed<false>& vocab)
	{
		// todo
	}

	void learn_embedding_from_file(cstlm::vocab_uncompressed<false>& vocab, std::string file_name)
	{
		sdsl::int_vector_buffer<0>			   text(file_name);
		std::uniform_int_distribution<int64_t> window_dist(1, m_window_size);
		std::uniform_int_distribution<int64_t> neg_table_dist(0, m_unigram_neg_table.size());
		std::mt19937						   gen(consts::RAND_SEED);

		sentence_parser<decltype(text.begin())> sentences(
		text.begin(), text.end(), vocab, m_sample_threshold, m_min_freq_threshold);

		std::vector<float> neu1(m_vector_size);
		std::vector<float> neu1e(m_vector_size);

		size_t last_sentences_processed = 0;
		size_t sentences_processed		= 0;
		double alpha					= m_start_learning_rate;
		size_t total_tokens				= std::distance(text.begin(), text.end());
		auto   start					= watch::now();
		while (sentences.next_sentence()) {
			// periodically update learning rate and output stats
			if (sentences_processed - last_sentences_processed > 1000) {
				float cur_pos	  = sentences.cur_offset_in_text();
				float text_percent = cur_pos / (float)(total_tokens + 1);
				auto  cur_time	 = watch::now();
				std::chrono::duration<double, std::ratio<1, 1>> secs_elapsed = cur_time - start;
				double sents_per_sec = double(sentences_processed) / (secs_elapsed.count() + 1);
				cstlm::LOG(cstlm::INFO) << "SENTS    " << sentences_processed << " Sentences "
										<< "SPEED    " << sents_per_sec << " Sentences/Sec "
										<< "PROGRESS " << text_percent << "%";

				// update learning rate based on how many words we have seen
				alpha = m_start_learning_rate * (1 - text_percent);
				if (alpha < m_start_learning_rate * 0.0001) alpha = m_start_learning_rate * 0.0001;

				last_sentences_processed = sentences_processed;
			}

			const auto& sentence = sentences.cur_sentence;
			if (sentences.cur_size() <= 1) continue;
			// we process each word in the sentence separately
			for (size_t sent_offset = 0; sent_offset < sentences.cur_size(); sent_offset++) {
				if (m_use_cbow) { // cbow

				} else { // skip-gram
					// examine a window around the current word
					// there is random window shrinkage in the original code
					// [xxxxxWxxxxx] -> [xxxWxxx]
					int64_t reduced_window_size				   = window_dist(gen);
					int64_t start							   = sent_offset - reduced_window_size;
					int64_t stop							   = sent_offset + reduced_window_size;
					if (stop >= (int64_t)sentence.size()) stop = sentence.size() - 1;
					for (int64_t wo = std::max(0L, start); wo < stop; wo++) {
						if (wo == (int64_t)sent_offset) continue; // don't examine the word itself

						auto word_id	= sentence[wo];
						auto row_offset = word_id * m_vector_size;
						for (size_t i = 0; i < neu1.size(); i++)
							neu1[i]   = 0;

						if (m_hierarchical_softmax) {
							// TODO
						} else {
							// negative sampling

							// (1) fill up the target vector with the
							// real target at pos 0 and the negative samples after
							for (size_t d = 0; d < m_num_negative_samples + 1; d++) {
								float label  = 1.0f;
								auto  target = word_id;
								if (d != 0) { // incorrect word. choose a negative sample at random
									// but keep the original word dist in mind
									target = m_unigram_neg_table[neg_table_dist(gen)];
									label  = 0.0f;
								}
								auto  l2 = target * m_vector_size;
								float f  = 0;
								for (size_t i = 0; i < neu1.size(); i++) {
									f += neu1[i] * m_net_state.syn1neg[i + l2];
								}
								// compute exp with hacks
								// 'g' is the gradient multiplied by the learning rate
								float g = 0;
								if (f > consts::MAX_EXP)
									g = (label - 1) * alpha;
								else if (f < -consts::MAX_EXP)
									g = (label - 0) * alpha;
								else {
									g = (label - m_expTable[(int)((f + consts::MAX_EXP) *
																  (consts::EXP_TABLE_SIZE /
																   consts::MAX_EXP / 2))]) *
										alpha;
								}
								for (size_t i = 0; i < neu1.size(); i++) {
									neu1e[i] += g * m_net_state.syn1neg[i + l2];
									m_net_state.syn1neg[i + l2] +=
									g * m_net_state.syn0[wo + row_offset];
								}
							}
						}
						// Learn weights input -> hidden
						for (size_t i = 0; i < neu1.size(); i++) {
							m_net_state.syn0[i + row_offset] += neu1e[i];
						}
					}
				}
			}
			sentences_processed++;
		}
	}

	embeddings extract_embeddings()
	{
		embeddings e;

		return e;
	}

	embeddings learn_embedding(cstlm::collection& col)
	{
		cstlm::vocab_uncompressed<false> vocab(col);

		// (1) initialize net state
		init_net_state(vocab);

		// (2) init unigram table for negative sampling
		if (m_num_negative_samples > 0) {
			init_unigram_table(vocab);
		}

		// (3) create huffman tree for hierarchical softmax
		if (m_hierarchical_softmax == true) {
			create_huffman_tree(vocab);
		}

		// (4) first train on the big file
		learn_embedding_from_file(vocab, col.file_map[cstlm::KEY_TEXT]);

		// (5) then train on the small file
		learn_embedding_from_file(vocab, col.file_map[cstlm::KEY_SMALLTEXT]);

		return extract_embeddings();
	}


public:
	builder& vector_size(uint32_t v)
	{
		m_vector_size = v;
		return *this;
	};

	builder& window_size(uint32_t ws)
	{
		m_window_size = ws;
		return *this;
	};

	builder& sample_threadhold(float st)
	{
		m_sample_threshold = st;
		return *this;
	};

	builder& num_negative_samples(uint32_t ns)
	{
		m_num_negative_samples = ns;
		if (ns == 0) {
			m_hierarchical_softmax = true;
		}
		return *this;
	};

	builder& num_iterations(uint32_t ni)
	{
		m_num_iterations = ni;
		return *this;
	};

	builder& num_threads(uint32_t nt)
	{
		m_num_threads = nt;
		return *this;
	};

	builder& min_freq_threshold(uint32_t ft)
	{
		m_min_freq_threshold = ft;
		return *this;
	};

	builder& start_learning_rate(float alpha)
	{
		m_start_learning_rate = alpha;
		return *this;
	};

	builder& batch_size(uint32_t bs)
	{
		m_batch_size = bs;
		return *this;
	};

	builder& use_cbow(bool cbow)
	{
		m_use_cbow = cbow;
		return *this;
	};

	builder& use_hierarchical_softmax(bool hs)
	{
		m_hierarchical_softmax = hs;
		if (m_hierarchical_softmax) {
			m_num_negative_samples = 0;
		}
		return *this;
	};

	embeddings train_or_load(cstlm::collection& col)
	{
		// (0) output params
		output_params();

		// (1) if exists. just load
		auto w2v_file = col.path + "/index/W2V.embeddings";
		if (cstlm::utils::file_exists(w2v_file)) {
			return embeddings(w2v_file);
		}

		// (2) train
		auto w2v_embeddings = learn_embedding(col);

		w2v_embeddings.store(w2v_file);

		return w2v_embeddings;
	}
};
}