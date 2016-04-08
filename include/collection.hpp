#pragma once

#include <stdexcept>
#include <chrono>
#include <sdsl/qsufsort.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>

#include "utils.hpp"
#include "logging.hpp"
#include "constants.hpp"
#include "timings.hpp"

namespace cstlm {

const std::string KEY_PREFIX = "text.";
const std::string KEY_PREFIX_BYTE = "text_byte.";
const std::string KEY_TEXT = "TEXT";
const std::string KEY_TEXTREV = "TEXTREV";
const std::string KEY_SA = "SA";
const std::string KEY_SAREV = "SAREV";
const std::string KEY_VOCAB = "VOCAB";
const std::string KEY_STATS = "STATS";

std::vector<std::string> collection_keys = { KEY_TEXT, KEY_TEXTREV, KEY_SA,
    KEY_SAREV, KEY_VOCAB };

enum class alphabet_type {
    byte_alphabet,
    word_alphabet,
    unknown_alphabet
};

struct collection {
    std::string path;
    uint64_t initial_vocab_size;
    uint64_t pruned_vocab_size;
    uint64_t num_non_freq_syms;
    uint64_t num_sentences;
    uint64_t num_tokens;
    uint64_t raw_size_in_bytes;
    uint64_t min_symbol_freq;
    std::string prefix;
    alphabet_type alphabet;

    std::map<std::string, std::string> file_map;
    collection() = default;
    collection(const std::string& p, alphabet_type a = alphabet_type::unknown_alphabet)
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

        /* make sure the necessary files are present */
        if (!utils::file_exists(path + "/" + prefix + KEY_TEXT)) {
            LOG(FATAL) << "collection path does not contain text.";
            throw std::runtime_error("collection path does not contain text.");
        }
        if (!utils::file_exists(path + "/" + prefix + KEY_VOCAB)) {
            LOG(FATAL) << "collection path does not contain vocabulary.";
            throw std::runtime_error("collection path does not contain vocabulary.");
        }
        /* register files that are present */
        for (const auto& key : collection_keys) {
            auto file_path = path + "/" + prefix + key;
            if (utils::file_exists(file_path)) {
                file_map[key] = file_path;
                LOG(INFO) << "FOUND '" << key << "' at '" << file_path << "'";
            }
        }
        if (file_map.count(KEY_SA) == 0) {
            lm_construct_timer timer(KEY_SA);
            if (alphabet == alphabet_type::byte_alphabet) {
                sdsl::int_vector<8> text;
                sdsl::load_from_file(text, file_map[KEY_TEXT].c_str());
                sdsl::int_vector<> sa;
                sa.width(64);
                sa.resize(text.size());
                divsufsort64((const unsigned char*)text.data(), (int64_t*)sa.data(), text.size());
                sdsl::util::bit_compress(sa);
                auto sa_path = path + "/" + prefix + KEY_SA;
                sdsl::store_to_file(sa, sa_path);
                file_map[KEY_SA] = sa_path;
            }
            else {
                sdsl::int_vector<> sa;
                sdsl::qsufsort::construct_sa(sa, file_map[KEY_TEXT].c_str(), 0);
                auto sa_path = path + "/" + prefix + KEY_SA;
                sdsl::store_to_file(sa, sa_path);
                file_map[KEY_SA] = sa_path;
            }
        }
        auto stats_file = path + "/" + prefix + KEY_STATS;
        if (utils::file_exists(stats_file)) {
            std::ifstream ifs(path + "/" + prefix + KEY_STATS);
            std::string s;
            while (std::getline(ifs, s)) {
                if (s.find("initial_vocab_size=") != std::string::npos) {
                    auto pos = s.find("=");
                    auto val = s.substr(pos + 1);
                    initial_vocab_size = std::strtoull(val.c_str(), NULL, 10);
                }
                if (s.find("pruned_vocab_size=") != std::string::npos) {
                    auto pos = s.find("=");
                    auto val = s.substr(pos + 1);
                    pruned_vocab_size = std::strtoull(val.c_str(), NULL, 10);
                }
                if (s.find("num_non_freq_syms=") != std::string::npos) {
                    auto pos = s.find("=");
                    auto val = s.substr(pos + 1);
                    num_non_freq_syms = std::strtoull(val.c_str(), NULL, 10);
                }
                if (s.find("num_sentences=") != std::string::npos) {
                    auto pos = s.find("=");
                    auto val = s.substr(pos + 1);
                    num_sentences = std::strtoull(val.c_str(), NULL, 10);
                }
                if (s.find("num_tokens=") != std::string::npos) {
                    auto pos = s.find("=");
                    auto val = s.substr(pos + 1);
                    num_tokens = std::strtoull(val.c_str(), NULL, 10);
                }
                if (s.find("raw_size_in_bytes=") != std::string::npos) {
                    auto pos = s.find("=");
                    auto val = s.substr(pos + 1);
                    raw_size_in_bytes = std::strtoull(val.c_str(), NULL, 10);
                }
                if (s.find("min_symbol_freq=") != std::string::npos) {
                    auto pos = s.find("=");
                    auto val = s.substr(pos + 1);
                    min_symbol_freq = std::strtoull(val.c_str(), NULL, 10);
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
        // use files available instead
        if (utils::file_exists(path + "/" + KEY_PREFIX_BYTE + KEY_TEXT)) {
            prefix = KEY_PREFIX_BYTE;
            return alphabet_type::byte_alphabet;
        }
        if (utils::file_exists(path + "/" + KEY_PREFIX + KEY_TEXT)) {
            prefix = KEY_PREFIX;
            return alphabet_type::word_alphabet;
        }
        LOG(FATAL) << "could not determine alphabet type. invalid collection dir?";
        return alphabet_type::word_alphabet;
    }

    std::string temp_file(std::string id)
    {
        return path + "/tmp/" + id + "-" + std::to_string(sdsl::util::pid()) + ".sdsl";
    }
};
}