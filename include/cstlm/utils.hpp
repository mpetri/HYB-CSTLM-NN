#pragma once

#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


#include "logging.hpp"
#include "timings.hpp"

#include <sys/resource.h>

namespace cstlm {

static int num_cstlm_threads = 0;

namespace utils {

std::string clean_word(const char* buf, size_t n)
{
    // check if the word contains non-ascii symbols -> skip
    for (size_t i = 0; i < n; i++) {
        int sym = buf[i];
        if (!isascii(sym)) return "";
    }
    std::string word;
    // skip garbage at front
    size_t offset = 0;
    while (offset < n) {
        int sym = buf[offset];
        if (isalnum(sym)) {
            break;
        } else {
            offset++;
        }
    }
    size_t word_start = offset;

    // remove garbage from the end
    size_t last = n - 1;
    while (word_start < last) {
        int sym = buf[last];
        if (!isalnum(sym)) {
            last--;
        } else {
            break;
        }
    }

    if (word_start >= last) return "";
    size_t word_len = last - word_start + 1;

    // don't want single symbol words that are not alphanumerical
    if (word_len == 1) {
        int sym = buf[word_start];
        if (!isalnum(sym)) return "";
    }
    return std::string(buf + word_start, word_len);
}


std::vector<std::string> parse_line(const std::string& line, bool byte)
{
    std::vector<std::string> line_tokens;
    if (byte) {
        for (const auto& chr : line) {
            line_tokens.push_back(std::string(1, chr));
        }
    } else {
        char   tmp_buf[10000] = {0};
        size_t cur            = 0;
        for (size_t i = 0; i < line.size(); i++) {
            int sym = line[i];
            if (isspace(sym) && cur != 0) {
                auto word = clean_word(tmp_buf, cur);
                if (word.size() > 0) {
                    line_tokens.push_back(word);
                }
                cur = 0;
            } else {
                tmp_buf[cur++] = sym;
            }
        }
        if (cur) {
            auto word = clean_word(tmp_buf, cur);
            if (word.size() > 0) {
                line_tokens.push_back(word);
            }
        }
    }
    return line_tokens;
}

bool directory_exists(std::string dir)
{
    struct stat sb;
    const char* pathname = dir.c_str();
    if (stat(pathname, &sb) == 0 && (S_IFDIR & sb.st_mode)) {
        return true;
    }
    return false;
}

bool file_exists(std::string file_name)
{
    std::ifstream in(file_name);
    if (in) {
        in.close();
        return true;
    }
    return false;
}

void create_directory(std::string dir)
{
    if (!directory_exists(dir)) {
        if (mkdir(dir.c_str(), 0777) == -1) {
            LOG(FATAL) << "could not create directory";
        }
    }
}

size_t getPeakRSS()
{
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return (size_t)(rusage.ru_maxrss * 1024L);
}
}
}

namespace std {
template <>
struct hash<std::vector<uint64_t>> {
    typedef std::vector<uint64_t> argument_type;
    typedef std::size_t           result_type;
    result_type operator()(argument_type const& s) const
    {
        std::size_t seed = 0;
        for (auto& i : s) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
}
