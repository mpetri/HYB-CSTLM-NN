#pragma once

#include "utils.hpp"

struct sentence_eval {
    double logprob;
    double tokens;
    sentence_eval(double lp, size_t t) : logprob(lp), tokens(t) {}
};


struct word_token {
    uint32_t small_id;
    uint32_t big_id;
    word_token(uint32_t s, uint32_t b) : small_id(s), big_id(b) {}
};


struct sentence_parser {

    static std::vector<std::vector<word_token>> parse(std::string file_name,
                                                      const cstlm::vocab_uncompressed<false>& vocab)
    {
        std::vector<std::vector<word_token>> sentences;
        sdsl::int_vector_buffer<0>           text(file_name);

        std::vector<word_token> cur;
        bool                    in_sentence = false;
        for (size_t i = 0; i < text.size(); i++) {
            auto sym = text[i];
            if (in_sentence == false && sym != cstlm::PAT_START_SYM) continue;

            if (sym == cstlm::PAT_START_SYM) {
                cur.emplace_back(sym, sym);
                in_sentence = true;
                continue;
            }

            if (sym == cstlm::PAT_END_SYM) {
                cur.emplace_back(sym, sym);
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
            cur.emplace_back(small_vocab_id, sym);
        }
        return sentences;
    }

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
            tokens.emplace_back(cstlm::PAT_START_SYM, cstlm::PAT_START_SYM);
            for (const auto& token : line_tokens) {
                auto big_id         = vocab.token2id(token, cstlm::UNKNOWN_SYM);
                auto small_vocab_id = filtered_vocab.big2small(big_id);
                tokens.emplace_back(small_vocab_id, big_id);
            }
            tokens.emplace_back(cstlm::PAT_END_SYM, cstlm::PAT_END_SYM);
            sentences.push_back(tokens);
        }
        return sentences;
    }
};
