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

const std::string KEY_PREFIX = "text.";
const std::string KEY_TEXT = "TEXT";
const std::string KEY_TEXTREV = "TEXTREV";
const std::string KEY_SA = "SA";
const std::string KEY_SAREV = "SAREV";
const std::string KEY_VOCAB = "VOCAB";

std::vector<std::string> collection_keys = { KEY_TEXT, KEY_TEXTREV, KEY_SA, KEY_SAREV, KEY_VOCAB };

struct collection {
    std::string path;
    std::map<std::string, std::string> file_map;
    collection() = default;
    collection(const std::string& p)
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
        /* make sure the necessary files are present */
        if (!utils::file_exists(path + "/" + KEY_PREFIX + KEY_TEXT)) {
            LOG(FATAL) << "collection path does not contain text.";
            throw std::runtime_error("collection path does not contain text.");
        }
        if (!utils::file_exists(path + "/" + KEY_PREFIX + KEY_VOCAB)) {
            LOG(FATAL) << "collection path does not contain vocabulary.";
            throw std::runtime_error("collection path does not contain vocabulary.");
        }
        /* register files that are present */
        for (const auto& key : collection_keys) {
            auto file_path = path + "/" + KEY_PREFIX + key;
            if (utils::file_exists(file_path)) {
                file_map[key] = file_path;
                LOG(INFO) << "FOUND '" << key << "' at '" << file_path << "'";
            }
        }
        if (file_map.count(KEY_SA) == 0) {
            utils::lm_mem_monitor::event("BUILD_SA");
            lm_construct_timer timer(KEY_SA);
            sdsl::int_vector<> sa;
            sdsl::qsufsort::construct_sa(sa, file_map[KEY_TEXT].c_str(), 0);
            auto sa_path = path + "/" + KEY_PREFIX + KEY_SA;
            sdsl::store_to_file(sa, sa_path);
            file_map[KEY_SA] = sa_path;
        }
    }
};
