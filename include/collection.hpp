#pragma once

#include <stdexcept>
#include <chrono>
#include <sdsl/qsufsort.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/int_vector_mapper.hpp>

#include "utils.hpp"

const std::string KEY_PREFIX = "text.";
const std::string KEY_TEXT = "TEXT";
const std::string KEY_TEXTREV = "TEXTREV";
const std::string KEY_SA = "SA";
const std::string KEY_SAREV = "SAREV";

std::vector<std::string> collection_keys = { KEY_TEXT,
                                             KEY_TEXTREV,
                                             KEY_SA,
                                             KEY_SAREV };

struct collection {
    std::string path;
    std::map<std::string, std::string> file_map;
    collection() = default;
    collection(const std::string& p,bool output = true)
        : path(p + "/")
    {
        using clock = std::chrono::high_resolution_clock;
        if (!utils::directory_exists(path)) {
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
            throw std::runtime_error("collection path does not contain text.");
        }
        /* register files that are present */
        for (const auto& key : collection_keys) {
            auto file_path = path + "/" + KEY_PREFIX + key;
            if (utils::file_exists(file_path)) {
                file_map[key] = file_path;
                if(output) std::cout << "FOUND '" << key << "' at '" << file_path << "'" << std::endl;
            }
        }
        /* create stuff we are missing */
        if (file_map.count(KEY_TEXTREV) == 0) {
            auto textrev_path = path + "/" + KEY_PREFIX + KEY_TEXTREV;
            if(output) std::cout << "CONSTRUCT " << KEY_TEXTREV << std::endl;
            auto start = clock::now();
            const sdsl::int_vector_mapper<0, std::ios_base::in> sdsl_input(file_map[KEY_TEXT]);
            {
                sdsl::int_vector<> tmp;
                std::ofstream ofs(textrev_path);
                sdsl::serialize(tmp, ofs);
            }
            sdsl::int_vector_mapper<0, std::ios_base::out | std::ios_base::in> sdsl_revinput(textrev_path);
            sdsl_revinput.resize(sdsl_input.size());
            // don't copy the last 0 and 1
            std::reverse_copy(std::begin(sdsl_input), std::end(sdsl_input) - 2, std::begin(sdsl_revinput));
            sdsl_revinput[sdsl_input.size() - 2] = 1ULL;
            sdsl_revinput[sdsl_input.size() - 1] = 0ULL;
            sdsl::util::bit_compress(sdsl_revinput);
            file_map[KEY_TEXTREV] = textrev_path;
            auto stop = clock::now();
            if(output) std::cout << "DONE ("
                      << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f
                      << ")" << std::endl;
        }

        if (file_map.count(KEY_SA) == 0) {
            if(output) std::cout << "CONSTRUCT " << KEY_SA << std::endl;
            auto start = clock::now();
            sdsl::int_vector<> sa;
            sdsl::qsufsort::construct_sa(sa, file_map[KEY_TEXT].c_str(), 0);
            auto sa_path = path + "/" + KEY_PREFIX + KEY_SA;
            sdsl::store_to_file(sa, sa_path);
            file_map[KEY_SA] = sa_path;
            auto stop = clock::now();
            if(output) std::cout << "DONE ("
                      << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f
                      << ")" << std::endl;
        }

        if (file_map.count(KEY_SAREV) == 0) {
            if(output) std::cout << "CONSTRUCT " << KEY_SAREV << std::endl;
            auto start = clock::now();
            sdsl::int_vector<> sarev;
            sdsl::qsufsort::construct_sa(sarev, file_map[KEY_TEXTREV].c_str(), 0);
            auto sarev_path = path + "/" + KEY_PREFIX + KEY_SAREV;
            sdsl::store_to_file(sarev, sarev_path);
            file_map[KEY_SAREV] = sarev_path;
            auto stop = clock::now();
            if(output) std::cout << "DONE ("
                      << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0f
                      << ")" << std::endl;
        }
    }
};