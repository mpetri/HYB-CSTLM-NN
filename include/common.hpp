#pragma once

#include "utils.hpp"

struct sentence_eval {
    double logprob;
    double tokens;
    sentence_eval(double lp, size_t t) : logprob(lp), tokens(t) {}
};


struct word_token {
    uint32_t    small_id;
    uint32_t    big_id;
    std::string tok_str;
    bool        is_oov;
    word_token(uint32_t s, uint32_t b, std::string ts, bool oov)
        : small_id(s), big_id(b), tok_str(ts), is_oov(oov)
    {
    }
};

std::ostream& operator<<(std::ostream& stream, const word_token& tok)
{
    stream << "<" << small_id << "," << big_id << ",'" << tok_str << "'," << is_oov << ">";
    return stream;
}

struct sentence_parser {

    static std::vector<std::vector<word_token>>
    parse_from_raw(std::string                             file_name,
                   const cstlm::vocab_uncompressed<false>& vocab,
                   const cstlm::vocab_uncompressed<false>& filtered_vocab)
    {
        std::vector<std::vector<word_token>> sentences;
        std::ifstream                        ifile(file_name);
        std::string                          line;
        while (std::getline(ifile, line)) {
            auto                    line_tokens = cstlm::utils::parse_line(line, false);
            std::vector<word_token> tokens;
            tokens.emplace_back(cstlm::PAT_START_SYM, cstlm::PAT_START_SYM, "<S>", false);
            for (const auto& token : line_tokens) {
                auto big_id         = vocab.token2id(token, cstlm::UNKNOWN_SYM);
                auto small_vocab_id = filtered_vocab.big2small(big_id);
                bool is_oov =
                (big_id == cstlm::UNKNOWN_SYM) && (small_vocab_id == cstlm::UNKNOWN_SYM);
                tokens.emplace_back(small_vocab_id, big_id, token, is_oov);
            }
            tokens.emplace_back(cstlm::PAT_END_SYM, cstlm::PAT_END_SYM, "</S>", false);
            sentences.push_back(tokens);
        }
        return sentences;
    }
};
