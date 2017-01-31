#pragma once

#include "vocab_uncompressed.hpp"
#include "collection.hpp"
#include "logging.hpp"

#include <chrono>
#include <atomic>

#include "omp.h"

#include <Eigen/Dense>
#include <sdsl/bit_vectors.hpp>

using namespace std::chrono;
using watch = std::chrono::high_resolution_clock;

namespace word2vec {

namespace consts {
const uint64_t RAND_SEED		  = 0XBEEF;
const uint64_t EXP_TABLE_SIZE	 = 1000;
const int64_t  MAX_EXP			  = 6;
const uint64_t MAX_SENTENCE_LEN   = 1000;
const uint64_t UNIGRAM_TABLE_SIZE = 1e8;
}

namespace constants {

const uint32_t DEFAULT_VEC_SIZE			= 100;
const float	DEFAULT_LEARNING_RATE	= 0.025f;
const uint32_t DEFAULT_WINDOW_SIZE		= 5;
const float	DEFAULT_SAMPLE_THRESHOLD = 1e-5;
const uint32_t DEFAULT_NUM_NEG_SAMPLES  = 5;
const uint32_t DEFAULT_NUM_ITERATIONS   = 5;
const uint32_t DEFAULT_MIN_FREQ_THRES   = 5;
}

struct embeddings {
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> data;
	embeddings() {}
	embeddings(std::string file_name) { load_binary(file_name); }

	void load_binary(std::string file_name)
	{
		FILE* f = fopen(file_name.c_str(), "r");
		if (f) {
			int64_t vocab_size;
			int64_t vector_size;
			if (2 != fscanf(f, "%ld %ld\n", &vocab_size, &vector_size)) {
				throw std::runtime_error("Error reading w2v embeddings.");
			}
			data.resize(vocab_size, vector_size);
			for (int64_t i = 0; i < data.rows(); i++) {
				/* read the word and discard */
				while (fgetc(f) != ' ') {
				}

				for (int64_t j = 0; j < data.cols(); j++) {
					float num = 0;
					if (1 != fread(&num, sizeof(num), 1, f)) {
						throw std::runtime_error("Error reading w2v embeddings.");
					}
					data(i, j) = num;
				}
			}
		} else {
			throw std::runtime_error("Cannot open file to read embeddings.");
		}
	}

	void load_plain(std::string file_name)
	{
		FILE* f = fopen(file_name.c_str(), "r");
		if (f) {
			int64_t vocab_size;
			int64_t vector_size;
			if (2 != fscanf(f, "%ld %ld\n", &vocab_size, &vector_size)) {
				throw std::runtime_error("Error reading w2v embeddings.");
			}
			data.resize(vocab_size, vector_size);
			for (int64_t i = 0; i < data.rows(); i++) {
				/* read the word and discard */
				while (fgetc(f) != ' ') {
				}

				for (int64_t j = 0; j < data.cols(); j++) {
					float num = 0;
					if (1 != fscanf(f, "%f ", &num)) {
						throw std::runtime_error("Error reading w2v embeddings.");
					}
					data(i, j) = num;
				}
			}
		} else {
			throw std::runtime_error("Cannot open file to read embeddings.");
		}
	}

	void store_binary(std::string file_name, cstlm::vocab_uncompressed<false>& vocab)
	{
		FILE* f = fopen(file_name.c_str(), "w");
		if (f) {
			fprintf(f, "%ld %ld\n", data.rows(), data.cols());
			for (int64_t i = 0; i < data.rows(); i++) {
				auto word = vocab.id2token(i);
				fprintf(f, "%s ", word.c_str());
				for (int64_t j = 0; j < data.cols(); j++) {
					float num = data(i, j);
					fwrite(&num, sizeof(float), 1, f);
				}
				fprintf(f, "\n");
			}
			fclose(f);
		} else {
			throw std::runtime_error("Cannot open file to write embeddings.");
		}
	}

	void store_plain(std::string file_name, cstlm::vocab_uncompressed<false>& vocab)
	{
		FILE* f = fopen(file_name.c_str(), "w");
		if (f) {
			fprintf(f, "%ld %ld\n", data.rows(), data.cols());
			for (int64_t i = 0; i < data.rows(); i++) {
				auto word = vocab.id2token(i);
				fprintf(f, "%s ", word.c_str());
				for (int64_t j = 0; j < data.cols(); j++) {
					float num = data(i, j);
					fprintf(f, "%f ", num);
				}
				fprintf(f, "\n");
			}
			fclose(f);
		} else {
			throw std::runtime_error("Cannot open file to write embeddings.");
		}
	}
};

struct net_state {
	std::vector<float> syn0;
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

	void print_cur(size_t i)
	{
		fprintf(stdout, "S[%u] n=%d [", i, m_cur_sentence_len);
		for (size_t i = 0; i < m_cur_sentence_len; i++) {
			auto word = vocab.id2token(m_sentence_buf[i]);
			fprintf(stdout, "'%s',", word.c_str());
		}
		fprintf(stdout, "]\n");
	}

	bool next_sentence()
	{
		size_t offset = 0;
		// find start of next sentence
		while (cur != end) {
			auto sym = *cur;
			++cur;
			if (sym == cstlm::PAT_START_SYM) {
				break;
			}
		}
		// process symbols until the end of the sentence
		while (cur != end) {
			auto sym	 = *cur;
			bool add_sym = true;
			if (sym == cstlm::PAT_END_SYM) { // sentence start sym found
				break;
			}
			if (offset == consts::MAX_SENTENCE_LEN) {
				break;
			}
			double freq = vocab.freq[sym];
			// drop words below min-freq
			if (freq < min_freq) {
				add_sym = false;
			} else {
				// The subsampling randomly discards frequent words while keeping the ranking same
				if (sample_threshold > 0) {
					double cprob = freq / double(vocab.total_freq);
					auto   prob  = 1.0 - sqrt(sample_threshold / cprob);
					auto   gprob = subsampling_dist(gen);
					auto   word  = vocab.id2token(sym);
					if (prob > gprob) {
						add_sym = false;
						// fprintf(stdout, " -> DROP\n");
					} else {
						// fprintf(stdout, " -> KEEP\n");
					}
				}
			}

			if (add_sym) {
				m_sentence_buf[offset++] = sym;
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
	uint32_t m_num_threads			= std::thread::hardware_concurrency();

private:
	net_state m_net_state;
	std::array<float, consts::EXP_TABLE_SIZE + 1> m_expTable;
	float								  m_cur_learning_rate;
	std::atomic<uint64_t>				  m_total_sentences_processed = ATOMIC_VAR_INIT(0);
	uint64_t							  m_total_tokens			  = 0;
	sdsl::bit_vector_il<256>			  m_unigram_bv;
	sdsl::bit_vector_il<256>::rank_1_type m_unigram_bv_rank;

private:
	void output_params()
	{
		cstlm::LOG(cstlm::INFO) << "W2V number of threads: " << m_num_threads;
		cstlm::LOG(cstlm::INFO) << "W2V number of iterations: " << m_num_iterations;
		cstlm::LOG(cstlm::INFO) << "W2V hidden size: " << m_vector_size;
		cstlm::LOG(cstlm::INFO) << "W2V number of negative samples: " << m_num_negative_samples;
		cstlm::LOG(cstlm::INFO) << "W2V window size: " << m_window_size;
		cstlm::LOG(cstlm::INFO) << "W2V sample threshold: " << m_sample_threshold;
		cstlm::LOG(cstlm::INFO) << "W2V min freq: " << m_min_freq_threshold;
		cstlm::LOG(cstlm::INFO) << "W2V starting learning rate: " << m_start_learning_rate;
		cstlm::LOG(cstlm::INFO) << "W2V model: skip-gram (SG)";
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

		if (m_num_negative_samples > 0) {
			m_net_state.syn1neg.resize(vocab.size() * m_vector_size);
			float* data		 = m_net_state.syn1neg.data();
			size_t data_size = m_net_state.syn1neg.size();
#pragma omp		   simd
			for (size_t i = 0; i < data_size; i++) {
				data[i] = 0;
			}
		}
	}

	void init_unigram_table(cstlm::vocab_uncompressed<false>& vocab)
	{
		// from the paper -> power 3/4
		const float power			= 0.75f;
		double		train_words_pow = 0.0;
		for (size_t i = 0; i < vocab.size(); i++) {
			train_words_pow += pow(vocab.freq[i], power);
		}
		sdsl::bit_vector unigram_bv;
		unigram_bv.resize(train_words_pow + 1);
		for (size_t i = 0; i < unigram_bv.size(); i++) {
			unigram_bv[i] = 0;
		}

		size_t cur = 0;
		for (size_t i = 0; i < vocab.size() - 1; i++) {
			cur += pow(vocab.freq[i], power);
			unigram_bv[cur] = 1;
		}
		m_unigram_bv	  = sdsl::bit_vector_il<256>(unigram_bv);
		m_unigram_bv_rank = sdsl::bit_vector_il<256>::rank_1_type(&m_unigram_bv);
	}

	void learn_embedding_from_file_chunk(cstlm::vocab_uncompressed<false>& vocab,
										 std::string					   file_name,
										 size_t							   file_offset,
										 size_t							   file_chunk_len,
										 uint64_t						   thread_id,
										 uint64_t						   cur_iteration)
	{
		sdsl::int_vector_buffer<0>			   text(file_name);
		std::uniform_int_distribution<int64_t> window_dist(1, m_window_size);
		std::uniform_int_distribution<int64_t> neg_sample_dist(1, m_unigram_bv.size() - 1);
		std::mt19937						   gen(consts::RAND_SEED);

		auto itr = text.begin() + file_offset;
		auto end = text.begin() + file_offset + file_chunk_len;

		sentence_parser<decltype(text.begin())> sentences(
		itr, end, vocab, m_sample_threshold, m_min_freq_threshold);

		std::vector<float> neu1e_data(m_vector_size);

		// get float ptrs for SIMD parallelism
		auto neu1e   = neu1e_data.data();
		auto syn1neg = m_net_state.syn1neg.data();
		auto syn0	= m_net_state.syn0.data();

		size_t last_sentences_processed = 0;
		size_t sentences_processed		= 0;
		size_t total_tokens				= std::distance(itr, end);
		auto   start					= watch::now();
		auto   last_msg					= watch::now();
		while (sentences.next_sentence()) {
			const auto& sentence = sentences.cur_sentence;
			// sentences.print_cur(sentences_processed);

			// periodically update learning rate and output stats
			if (sentences_processed - last_sentences_processed > 2000) {
				float local_cur_pos = sentences.cur_offset_in_text();
				float cur_pos = sentences.cur_offset_in_text() + (cur_iteration * total_tokens);
				float text_percent = cur_pos / (float)(total_tokens * m_num_iterations);
				auto  cur_time	 = watch::now();
				std::chrono::duration<double, std::ratio<1, 1>> secs_elapsed = cur_time - start;
				double sents_per_sec = double(sentences_processed) / (secs_elapsed.count() + 1);
				double words_per_sec = double(local_cur_pos) / (secs_elapsed.count() + 1);

				std::chrono::duration<double, std::ratio<1, 1>> pelapsed = cur_time - last_msg;
				if (pelapsed.count() > 3) {
					cstlm::LOG(cstlm::INFO)
					<< "W2V [" << thread_id << "] "
					<< "iter(" << cur_iteration + 1 << "/" << m_num_iterations << ") "
					<< "alpha(" << std::fixed << std::setprecision(6) << m_cur_learning_rate << ") "
					<< "S(" << std::setprecision(0) << float(sentences_processed) << ") "
					<< "W(" << std::setprecision(0) << local_cur_pos << ") "
					<< "S/s(" << std::setprecision(0) << sents_per_sec << ") "
					<< "W/s(" << std::setprecision(0) << words_per_sec << ") "
					<< "%(" << std::setprecision(2) << floorf(text_percent * 10000.0f) / 100.0f
					<< ") ";
					last_msg = cur_time;
				}

				// update learning rate based on how many words we have seen
				float new_rate = m_start_learning_rate * ((1 - (text_percent)));
				if (new_rate < m_start_learning_rate * 0.0001)
					new_rate = m_start_learning_rate * 0.0001;

				m_cur_learning_rate = new_rate;

				last_sentences_processed = sentences_processed;
			}

			if (sentences.cur_size() <= 1) continue;
			// we process each word in the sentence separately
			for (size_t sent_offset = 0; sent_offset < sentences.cur_size(); sent_offset++) {
				// examine a window around the current word
				// there is random window shrinkage in the original code
				// [xxxxxWxxxxx] -> [xxxWxxx]
				int64_t reduced_window_size						= window_dist(gen);
				int64_t start									= sent_offset - reduced_window_size;
				int64_t stop									= sent_offset + reduced_window_size;
				auto	target_word								= sentence[sent_offset];
				if (start < 0) start							= 0;
				if (stop >= (int64_t)sentences.cur_size()) stop = sentences.cur_size() - 1;
				for (int64_t wo = start; wo < stop; wo++) {
					if (wo == (int64_t)sent_offset) continue; // don't examine the word itself

					auto word_id	= sentence[wo];
					auto row_offset = word_id * m_vector_size;

					{
#pragma omp simd
						for (size_t i = 0; i < m_vector_size; i++)
							neu1e[i]  = 0;
					}

					// negative sampling only!
					// (1) fill up the target vector with the
					// real target at pos 0 and the negative samples after
					for (size_t d = 0; d < m_num_negative_samples + 1; d++) {
						float label  = 1.0f;
						auto  target = target_word;
						if (d != 0) { // incorrect word. choose a negative sample at random
							// but keep the original word dist in mind
							target = m_unigram_bv_rank(neg_sample_dist(gen));
							// static std::mutex			m;
							// std::lock_guard<std::mutex> ml(m);
							// std::cout << "target = " << target
							// 		  << " vocab size = " << vocab.size() << std::endl;
							if (target >= vocab.size()) {
								std::cerr << "error neg generation -> " << target
										  << " vocab=" << vocab.size() << std::endl;
							}
							//target = m_unigram_neg_table[neg_table_dist(gen)];
							if (target == target_word) continue;
							label = 0.0f;
						}
						auto  l2 = target * m_vector_size;
						float f  = 0;
						for (size_t i = 0; i < m_vector_size; i++) {
							f += syn0[i + row_offset] * syn1neg[i + l2];
						}
						// compute exp with hacks
						// 'g' is the gradient multiplied by the learning rate
						float g = 0;
						if (f > float(consts::MAX_EXP)) {
							g = (label - 1) * m_cur_learning_rate;
						} else if (f < float(-consts::MAX_EXP)) {
							g = (label - 0) * m_cur_learning_rate;
						} else {
							const int co = consts::EXP_TABLE_SIZE / consts::MAX_EXP / 2;
							int		  o  = (f + consts::MAX_EXP) * co;
							g			 = (label - m_expTable[o]) * m_cur_learning_rate;
						}

						// TODO WHAT IS THIS
						{
#pragma omp simd
							for (size_t i = 0; i < m_vector_size; i++) {
								neu1e[i] += g * syn1neg[i + l2];
							}
						}

						// TODO WHAT IS THIS??
						{
#pragma omp simd
							for (size_t i = 0; i < m_vector_size; i++) {
								syn1neg[i + l2] += g * syn0[i + row_offset];
							}
						}
					}
					// Learn weights input -> hidden
					{
#pragma omp simd
						for (size_t i = 0; i < m_vector_size; i++) {
							syn0[i + row_offset] += neu1e[i];
						}
					}
				}
			}
			sentences_processed++;
		}

		if (cur_iteration == 0) {
			std::atomic_fetch_add(&m_total_sentences_processed, sentences_processed);
		}
	}


	void learn_embedding_from_file(cstlm::vocab_uncompressed<false>& vocab, std::string file_name)
	{
		auto text_size = 0;
		{
			sdsl::int_vector_buffer<0> text(file_name);
			text_size = text.size();
		}


		m_cur_learning_rate = m_start_learning_rate;
		auto start			= watch::now();
		for (size_t j = 0; j < m_num_iterations; j++) {
			auto						   coffset	= 0;
			auto						   chunk_size = text_size / m_num_threads;
			std::vector<std::future<void>> fchunks;
			for (size_t i = 0; i < m_num_threads; i++) {
				if (i + 1 == m_num_threads) { // last thread processes everything
					chunk_size = text_size - coffset;
				}
				fchunks.push_back(
				std::async(std::launch::async, [&, i, j, chunk_size, coffset, file_name] {
					learn_embedding_from_file_chunk(vocab, file_name, coffset, chunk_size, i, j);
				}));
				coffset += chunk_size;
			}
			for (size_t i = 0; i < m_num_threads; i++) {
				fchunks[i].wait();
			}
		}
		auto stop = watch::now();

		std::chrono::duration<double, std::ratio<1, 1>> secs_elapsed = stop - start;
		double sents_per_sec = double(m_total_sentences_processed) / (secs_elapsed.count() + 1);
		double words_per_sec = double(text_size) / (secs_elapsed.count() + 1);
		cstlm::LOG(cstlm::INFO) << "W2V Sentences Processed: " << m_total_sentences_processed << " "
								<< "Words Processed: " << text_size << " "
								<< "S/s(" << std::setprecision(0) << sents_per_sec << ") "
								<< "W/s(" << std::setprecision(0) << words_per_sec << ") "
								<< "Total Time: " << std::setprecision(2) << secs_elapsed.count()
								<< " secs";
		m_total_tokens += text_size;
	}

	embeddings extract_embeddings(cstlm::vocab_uncompressed<false>& vocab)
	{
		embeddings e;
		e.data.resize(vocab.size(), m_vector_size);
		for (size_t i = 0; i < vocab.size(); i++) {
			for (size_t j = 0; j < m_vector_size; j++) {
				e.data(i, j) = m_net_state.syn0[i * m_vector_size + j];
			}
		}
		return e;
	}

	embeddings learn_embedding(cstlm::vocab_uncompressed<false>& vocab, cstlm::collection& col)
	{

		// (1) initialize net state
		cstlm::LOG(cstlm::INFO) << "W2V init net state";
		init_net_state(vocab);

		// (2) init unigram table for negative sampling
		if (m_num_negative_samples > 0) {
			cstlm::LOG(cstlm::INFO) << "W2V init unigram table";
			init_unigram_table(vocab);
		}

		// (3) first train on the big file
		cstlm::LOG(cstlm::INFO) << "W2V learn embedding big file";
		auto start = watch::now();
		learn_embedding_from_file(vocab, col.file_map[cstlm::KEY_TEXT]);

		// (4) then train on the small file
		/*
		cstlm::LOG(cstlm::INFO) << "learn embedding small file";
		learn_embedding_from_file(vocab, col.file_map[cstlm::KEY_SMALLTEXT]);
        */

		auto stop = watch::now();

		std::chrono::duration<double, std::ratio<1, 1>> secs_elapsed = stop - start;
		double sents_per_sec = double(m_total_sentences_processed) / (secs_elapsed.count() + 1);
		double words_per_sec = double(m_total_tokens) / (secs_elapsed.count() + 1);
		cstlm::LOG(cstlm::INFO) << "W2V Sentences Processed: " << m_total_sentences_processed << " "
								<< "Words Processed: " << m_total_tokens << " "
								<< "S/s(" << std::setprecision(0) << sents_per_sec << ") "
								<< "W/s(" << std::setprecision(0) << words_per_sec << ") "
								<< "Total Time: " << std::setprecision(2) << secs_elapsed.count()
								<< " secs";

		return extract_embeddings(vocab);
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

	std::string file_name(cstlm::collection& col)
	{
		std::stringstream fn;
		fn << col.path;
		fn << "/index/W2V-";
		fn << "mf=" << m_min_freq_threshold << "-";
		fn << "ni=" << m_num_iterations << "-";
		fn << "ns=" << m_num_negative_samples << "-";
		fn << "ss=" << m_sample_threshold << "-";
		fn << "ws=" << m_window_size << "-";
		fn << "vs=" << m_vector_size;
		fn << ".embeddings";
		return fn.str();
	}

	embeddings train_or_load(cstlm::collection& col)
	{
		omp_set_num_threads(m_num_threads);

		// (0) output params
		output_params();

		// (1) if exists. just load
		auto w2v_file = file_name(col);
		if (cstlm::utils::file_exists(w2v_file)) {
			return embeddings(w2v_file);
		}

		// (2) load vocab
		cstlm::LOG(cstlm::INFO) << "create vocab";
		cstlm::vocab_uncompressed<false> vocab(col);

		// (3) train
		auto w2v_embeddings = learn_embedding(vocab, col);

		w2v_embeddings.store_binary(w2v_file, vocab);
		w2v_embeddings.store_plain(w2v_file + ".plain", vocab);

		return w2v_embeddings;
	}
};
}
