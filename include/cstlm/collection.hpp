#pragma once

#include <chrono>
#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>
#include <sdsl/qsufsort.hpp>
#include <stdexcept>

#include "constants.hpp"
#include "logging.hpp"
#include "timings.hpp"
#include "utils.hpp"

#include "parallel_sa_construct.hpp"

namespace cstlm {

const std::string KEY_PREFIX		   = "text.";
const std::string KEY_PREFIX_BYTE	  = "text_byte.";
const std::string KEY_BIG_TEXT		   = "BIG_TEXT";
const std::string KEY_BIG_TEXTREV	  = "BIG_TEXTREV";
const std::string KEY_SA			   = "SA";
const std::string KEY_SAREV			   = "SAREV";
const std::string KEY_VOCAB			   = "VOCAB";
const std::string KEY_STATS			   = "STATS";
const std::string KEY_SMALL_TEXT	   = "SMALL_TEXT";
const std::string KEY_SMALL_TEXT_REV   = "SMALL_TEXTREV";
const std::string KEY_COMBINED_TEXT	= "COMBINED_TEXT";
const std::string KEY_COMBINED_TEXTREV = "COMBINED_TEXTREV";

const std::string KEY_CSTLM_TEXT	= "CSTLM_TEXT";
const std::string KEY_CSTLM_TEXTREV = "CSTLM_TEXTREV";

std::vector<std::string> collection_keys = {KEY_BIG_TEXT,
											KEY_BIG_TEXTREV,
											KEY_COMBINED_TEXT,
											KEY_COMBINED_TEXTREV,
											KEY_SMALL_TEXT,
											KEY_SMALL_TEXT_REV,
											KEY_SA,
											KEY_SAREV,
											KEY_VOCAB};

enum class alphabet_type { byte_alphabet, word_alphabet, unknown_alphabet };

struct collection {
	std::string   path;
	uint64_t	  vocab_size;
	uint64_t	  big_num_sentences;
	uint64_t	  big_num_tokens;
	uint64_t	  big_raw_size_in_bytes;
	uint64_t	  small_num_sentences;
	uint64_t	  small_num_tokens;
	uint64_t	  small_raw_size_in_bytes;
	uint64_t	  combined_num_sentences;
	uint64_t	  combined_num_tokens;
	uint64_t	  combined_raw_size_in_bytes;
	std::string   prefix;
	alphabet_type alphabet;

	std::map<std::string, std::string> file_map;
	collection() = default;
	collection(const std::string& p,
			   alphabet_type	  a		 = alphabet_type::unknown_alphabet,
			   bool				  output = true)
		: path(p + "/")
	{
		if (!utils::directory_exists(path)) {
			LOG(FATAL) << "collection path not found.";
			throw std::runtime_error("collection path not found.");
		}
		// make sure all other dirs exist
		std::string index_directory = path + "/index/";
		utils::create_directory(index_directory);
		std::string tmp_directory = path + "/tmp/";
		utils::create_directory(tmp_directory);
		std::string results_directory = path + "/results/";
		utils::create_directory(results_directory);
		std::string patterns_directory = path + "/patterns/";
		utils::create_directory(patterns_directory);

		alphabet = determine_alphabet_type(a);

		/* register files that are present */
		for (const auto& key : collection_keys) {
			auto file_path = path + "/" + prefix + key;
			if (utils::file_exists(file_path)) {
				file_map[key] = file_path;
				if (output) LOG(INFO) << "FOUND '" << key << "' at '" << file_path << "'";
			}
		}
		auto stats_file = path + "/" + prefix + KEY_STATS;
		if (utils::file_exists(stats_file)) {
			std::ifstream ifs(path + "/" + prefix + KEY_STATS);
			std::string   s;
			while (std::getline(ifs, s)) {
				if (s.find("vocab_size=") != std::string::npos) {
					auto pos   = s.find("=");
					auto val   = s.substr(pos + 1);
					vocab_size = std::strtoull(val.c_str(), NULL, 10);
				}

				if (s.find("combined_num_sentences=") != std::string::npos) {
					auto pos			   = s.find("=");
					auto val			   = s.substr(pos + 1);
					combined_num_sentences = std::strtoull(val.c_str(), NULL, 10);
				}
				if (s.find("combined_num_tokens=") != std::string::npos) {
					auto pos			= s.find("=");
					auto val			= s.substr(pos + 1);
					combined_num_tokens = std::strtoull(val.c_str(), NULL, 10);
				}
				if (s.find("combined_raw_size_in_bytes=") != std::string::npos) {
					auto pos				   = s.find("=");
					auto val				   = s.substr(pos + 1);
					combined_raw_size_in_bytes = std::strtoull(val.c_str(), NULL, 10);
				}

				if (s.find("big_num_sentences=") != std::string::npos) {
					auto pos		  = s.find("=");
					auto val		  = s.substr(pos + 1);
					big_num_sentences = std::strtoull(val.c_str(), NULL, 10);
				}
				if (s.find("big_num_tokens=") != std::string::npos) {
					auto pos	   = s.find("=");
					auto val	   = s.substr(pos + 1);
					big_num_tokens = std::strtoull(val.c_str(), NULL, 10);
				}
				if (s.find("big_raw_size_in_bytes=") != std::string::npos) {
					auto pos			  = s.find("=");
					auto val			  = s.substr(pos + 1);
					big_raw_size_in_bytes = std::strtoull(val.c_str(), NULL, 10);
				}

				if (s.find("small_num_sentences=") != std::string::npos) {
					auto pos			= s.find("=");
					auto val			= s.substr(pos + 1);
					small_num_sentences = std::strtoull(val.c_str(), NULL, 10);
				}
				if (s.find("small_num_tokens=") != std::string::npos) {
					auto pos		 = s.find("=");
					auto val		 = s.substr(pos + 1);
					small_num_tokens = std::strtoull(val.c_str(), NULL, 10);
				}
				if (s.find("small_raw_size_in_bytes=") != std::string::npos) {
					auto pos				= s.find("=");
					auto val				= s.substr(pos + 1);
					small_raw_size_in_bytes = std::strtoull(val.c_str(), NULL, 10);
				}
			}
		}
	}

	alphabet_type determine_alphabet_type(alphabet_type a)
	{
		// predefined value?
		if (a == alphabet_type::byte_alphabet) {
			prefix = KEY_PREFIX_BYTE;
			return alphabet_type::byte_alphabet;
		}
		if (a == alphabet_type::word_alphabet) {
			prefix = KEY_PREFIX;
			return alphabet_type::word_alphabet;
		}
		prefix = KEY_PREFIX;
		return alphabet_type::word_alphabet;
	}

	std::string temp_file(std::string id, uint64_t thread = 0)
	{
		return path + "/tmp/" + id + "-" + std::to_string(thread) + "-" +
			   std::to_string(sdsl::util::pid()) + ".sdsl";
	}
};

void construct_SA(collection& col)
{
	auto sa_path = col.path + "/" + col.prefix + col.file_map[KEY_CSTLM_TEXT] + "." + KEY_SA;
	lm_construct_timer timer(KEY_SA);
	if (col.alphabet == alphabet_type::byte_alphabet) {
		sdsl::int_vector<8> text;
		sdsl::load_from_file(text, col.file_map[KEY_CSTLM_TEXT].c_str());
		sdsl::int_vector<> sa;
		sa.width(64);
		sa.resize(text.size());
		divsufsort64((const unsigned char*)text.data(), (int64_t*)sa.data(), text.size());
		sdsl::util::bit_compress(sa);
		sdsl::store_to_file(sa, sa_path);
		col.file_map[KEY_SA] = sa_path;
	} else {
		sdsl::int_vector<> sa;
		sdsl::qsufsort::construct_sa(sa, col.file_map[KEY_CSTLM_TEXT].c_str(), 0);
		sdsl::store_to_file(sa, sa_path);
		col.file_map[KEY_SA] = sa_path;
	}
}
}