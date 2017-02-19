#pragma once

#include <unordered_map>
#include <stdexcept>

namespace cstlm {

template <bool byte_alphabet>
class vocab_uncompressed {
public:
    typedef sdsl::int_vector<>::size_type size_type;


private:
    std::unordered_map<std::string, uint64_t> m_t2i;
    std::unordered_map<uint64_t, std::string> m_i2t;
    std::unordered_map<uint64_t, uint64_t>    m_b2s;
    std::unordered_map<uint64_t, uint64_t>    m_s2b;

public:
    std::vector<uint64_t> freq;
    uint64_t              total_freq = 0;

public:
    vocab_uncompressed() = default;
    vocab_uncompressed(collection& col)
    {
        auto          vocab_file = col.file_map[KEY_VOCAB];
        std::ifstream vfs(vocab_file);
        std::string   line;
        uint64_t      id = 0;
        while (std::getline(vfs, line)) {
            auto last_sep_pos = line.rfind(' ');
            auto sep_pos      = line.rfind(' ', last_sep_pos - 1);
            auto word         = line.substr(0, sep_pos);
            if (byte_alphabet) {
                try {
                    auto char_sym = std::stoul(word);
                    word          = std::string(1, char_sym);
                } catch (...) {
                    /* could not convert -> just use word */
                }
            }
            auto id_str_len = last_sep_pos - sep_pos;
            auto str_id     = line.substr(sep_pos, id_str_len);
            id              = std::strtoull(str_id.c_str(), NULL, 10);

            auto     str_freq = line.substr(last_sep_pos);
            uint64_t f        = std::strtoull(str_freq.c_str(), NULL, 10);

            m_t2i[word] = id;
            m_i2t[id]   = word;
            if (freq.size() < id + 1) {
                freq.resize(1024 + 2 * id);
            }
            freq[id] = f;
            total_freq += f;
        }
        freq.resize(id + 1);
    }

    size_type
    serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL, std::string name = "") const
    {
        sdsl::structure_tree_node* child =
        sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type            written_bytes = 0;
        std::vector<uint8_t> token_data;
        sdsl::int_vector<>   ids(m_i2t.size());
        size_t               i = 0;
        for (const auto& p : m_i2t) {
            const auto& token = p.second;
            std::copy(token.begin(), token.end(), std::back_inserter(token_data));
            token_data.push_back(0); // delim
            ids[i] = p.first;
            i++;
        }
        written_bytes += sdsl::serialize(token_data, out, child, "tokens");
        written_bytes += sdsl::serialize(ids, out, child, "ids");

	sdsl::int_vector<> b2s(m_b2s.size()*2);
	size_t cur = 0;
	for(const auto& item : m_b2s) {
		b2s[cur++] = item.first;
		b2s[cur++] = item.second;
	}
        written_bytes += sdsl::serialize(b2s, out, child, "b2s");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    auto begin() const -> decltype(m_t2i.begin()) { return m_t2i.begin(); }

    auto end() const -> decltype(m_t2i.end()) { return m_t2i.end(); }

    std::string id2token(const uint64_t& id) const
    {
        auto itr = m_i2t.find(id);
        if (itr == m_i2t.end()) {
            std::string msg = "Id '" + std::to_string(id) + "' lookup failed.";
            throw std::runtime_error(msg);
        }
        return itr->second;
    }

    uint64_t token2id(const std::string& token) const
    {
        auto itr = m_t2i.find(token);
        if (itr == m_t2i.end()) {
            throw std::runtime_error("Token lookup failed.");
        }
        return itr->second;
    }

    template <class t_pat_iter>
    std::vector<std::string> id2token(t_pat_iter begin, t_pat_iter end) const
    {
        std::vector<std::string> ret;
        for (t_pat_iter it = begin; it != end; ++it)
            ret.push_back(id2token(*it));
        return ret;
    }

    uint64_t token2id(const std::string& token, uint64_t defaul) const
    {
        auto itr = m_t2i.find(token);
        if (itr == m_t2i.end()) {
            return defaul;
        }
        return itr->second;
    }

    void load(std::istream& in)
    {
        std::vector<uint8_t> token_data;
        sdsl::int_vector<>   ids;
        sdsl::load(token_data, in);
        sdsl::load(ids, in);

        auto tok_itr = token_data.begin();
        for (size_t i = 0; i < ids.size(); i++) {
            const auto& id      = ids[i];
            auto        sep_pos = std::find(tok_itr, std::end(token_data), 0);
            std::string tok(tok_itr, sep_pos);
            tok_itr = sep_pos + 1;
            m_i2t.emplace(id, tok);
            m_t2i.emplace(tok, id);
        }
	sdsl::int_vector<> b2s;
	sdsl::load(b2s,in);
	for(size_t i=0;i<b2s.size();i+=2) {
	     auto big = b2s[i];
	     auto small = b2s[i+1];
	     m_b2s[big] = small;
	     m_s2b[small] = big;
	}
    }

    void swap(vocab_uncompressed& a)
    {
        if (this != &a) {
            m_i2t.swap(a.m_i2t);
            m_t2i.swap(a.m_t2i);
            m_b2s.swap(a.m_b2s);
            m_s2b.swap(a.m_s2b);
        }
    }

    size_type size() const { return m_i2t.size(); }


    vocab_uncompressed filter(std::string input_file, uint32_t threshold) const
    {
        // (1) count frequencies
        sdsl::int_vector_buffer<0> text(input_file);
        using p_t = std::pair<uint32_t, uint32_t>;
        std::vector<p_t> counts(size());
        for (size_t i       = 0; i < counts.size(); i++)
            counts[i].first = i;
        for (size_t i = 0; i < text.size(); i++) {
            counts[text[i]].second++;
        }
        // (2) sort frequencies
        std::sort(counts.begin(), counts.end(), [](const p_t& a, const p_t& b) {
            return a.second > b.second;
        });

        // (3) create the filtered vocab
        vocab_uncompressed filtered_vocab;

        // (3a) add special symbols
        auto i2t = m_i2t;
        for (size_t i = 0; i < NUM_SPECIAL_SYMS; i++) {
            auto tok                  = i2t[i];
            filtered_vocab.m_t2i[tok] = i;
            filtered_vocab.m_i2t[i]   = tok;
            filtered_vocab.m_b2s[i]   = i;
            filtered_vocab.m_s2b[i]   = i;
        }

        // (3b) add threshold most frequent symbols in text
        size_t idx = 0;
        while (filtered_vocab.size() < threshold && counts.size() > idx) {
            auto cur           = counts[idx++];
            auto cur_big_id    = cur.first;
            auto next_small_id = filtered_vocab.size();
            if (cur_big_id >= NUM_SPECIAL_SYMS) {
                auto tok                            = i2t[cur_big_id];
                filtered_vocab.m_t2i[tok]           = next_small_id;
                filtered_vocab.m_i2t[next_small_id] = tok;
                filtered_vocab.m_b2s[cur_big_id]    = next_small_id;
                filtered_vocab.m_s2b[next_small_id] = cur_big_id;
            }
        }

        return filtered_vocab;
    }

    uint64_t big2small(uint64_t id, uint64_t defaul = UNKNOWN_SYM) const
    {
        auto itr = m_b2s.find(id);
        if (itr == m_b2s.end()) {
            return defaul;
        }
        return itr->second;
    }

    uint64_t small2big(uint64_t id, uint64_t defaul = UNKNOWN_SYM) const
    {
        auto itr = m_s2b.find(id);
        if (itr == m_s2b.end()) {
            return defaul;
        }
        return itr->second;
    }
};
}
