#pragma once

#include <unordered_map>
#include <stdexcept>

class vocab_uncompressed {
public:
    typedef sdsl::int_vector<>::size_type size_type;

private:
    std::unordered_map<std::string, uint64_t> m_t2i;
    std::unordered_map<uint64_t, std::string> m_i2t;

public:
    vocab_uncompressed() = default;
    vocab_uncompressed(collection& col)
    {
        auto vocab_file = col.file_map[KEY_VOCAB];
        std::ifstream vfs(vocab_file);
        std::string line;
        while (std::getline(vfs, line)) {
            auto sep_pos = line.rfind(' ');
            auto word = line.substr(0, sep_pos);
            auto str_id = line.substr(sep_pos);
            uint64_t id = std::strtoull(str_id.c_str(), NULL, 10);
            m_t2i[word] = id;
            m_i2t[id] = word;
        }
    }

    size_type serialize(std::ostream& out, sdsl::structure_tree_node* v = NULL,
                        std::string name = "") const
    {
        sdsl::structure_tree_node* child
            = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
        size_type written_bytes = 0;
        std::vector<uint8_t> token_data;
        sdsl::int_vector<> ids(m_i2t.size());
        size_t i = 0;
        for (const auto& p : m_i2t) {
            const auto& token = p.second;
            std::copy(token.begin(), token.end(), std::back_inserter(token_data));
            token_data.push_back(0); // delim
            ids[i] = p.first;
            i++;
        }
        written_bytes += sdsl::serialize(token_data, out, child, "tokens");
        written_bytes += sdsl::serialize(ids, out, child, "ids");
        sdsl::structure_tree::add_size(child, written_bytes);
        return written_bytes;
    }

    std::string id2token(const uint64_t& id) const
    {
        auto itr = m_i2t.find(id);
        if (itr == m_i2t.end()) {
            throw std::runtime_error("Id lookup failed.");
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
        sdsl::int_vector<> ids;
        sdsl::load(token_data, in);
        sdsl::load(ids, in);

        auto tok_itr = token_data.begin();
        for (size_t i = 0; i < ids.size(); i++) {
            const auto& id = ids[i];
            auto sep_pos = std::find(tok_itr, std::end(token_data), 0);
            std::string tok(tok_itr, sep_pos);
            tok_itr = sep_pos + 1;
            m_i2t.emplace(id, tok);
            m_t2i.emplace(tok, id);
        }
    }

    void swap(vocab_uncompressed& a)
    {
        if (this != &a) {
            m_i2t.swap(a.m_i2t);
            m_t2i.swap(a.m_t2i);
        }
    }

    size_type size() const { return m_i2t.size(); }
};
