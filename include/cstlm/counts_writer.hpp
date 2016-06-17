#pragma once

#include "sdsl/int_vector_buffer.hpp"
#include "collection.hpp"

namespace cstlm {

struct computed_context_result {
    uint64_t node_id;
    uint32_t f1;
    uint32_t f2;
    uint32_t fb;
    uint32_t b;
    uint32_t f1prime;
    uint32_t f2prime;
};

struct counts_writer {
    counts_writer(size_t id, collection& col)
        : m_id(id)
    {
        f1 = sdsl::int_vector_buffer<32>(col.temp_file("counts_f1", id), std::ios::out);
        f2 = sdsl::int_vector_buffer<32>(col.temp_file("counts_f2", id), std::ios::out);
        fb = sdsl::int_vector_buffer<32>(col.temp_file("counts_fb", id), std::ios::out);
        b = sdsl::int_vector_buffer<32>(col.temp_file("counts_b", id), std::ios::out);
        f1p = sdsl::int_vector_buffer<32>(col.temp_file("counts_f1p", id), std::ios::out);
        f2p = sdsl::int_vector_buffer<32>(col.temp_file("counts_f2p", id), std::ios::out);
        ids = sdsl::int_vector_buffer<64>(col.temp_file("node_ids", id), std::ios::out);
    }

    void write_result(computed_context_result& res)
    {
        f1.push_back(res.f1);
        f2.push_back(res.f2);
        fb.push_back(res.fb);
        b.push_back(res.b);
        f1p.push_back(res.f1prime);
        f2p.push_back(res.f2prime);
        ids.push_back(res.node_id);
    }

    void delete_temp_files()
    {
        f1.close(true);
        f2.close(true);
        fb.close(true);
        b.close(true);
        f1p.close(true);
        f2p.close(true);
        ids.close(true);
    }
    size_t m_id;
    sdsl::int_vector_buffer<32> f1;
    sdsl::int_vector_buffer<32> f2;
    sdsl::int_vector_buffer<32> fb;
    sdsl::int_vector_buffer<32> b;
    sdsl::int_vector_buffer<32> f1p;
    sdsl::int_vector_buffer<32> f2p;
    sdsl::int_vector_buffer<64> ids;
};

struct dummy_container {
    using size_type = uint64_t;
    using value_type = uint32_t;
    std::vector<sdsl::int_vector_buffer<32>*> bufs;
    size_type m_size = 0;
    dummy_container() = default;
    dummy_container(std::vector<sdsl::int_vector_buffer<32>*> b)
        : bufs(b)
    {
        m_size = 0;
        for (auto& buffer : bufs) {
            m_size += buffer->size();
        }
    }
    size_type size() const
    {
        return m_size;
    }

    value_type operator[](size_type idx) const
    {
        size_type cumsum = 0;
        for (size_t i = 0; i < bufs.size(); i++) {
            if (cumsum + bufs[i]->size() > idx) {
                size_t relidx = idx - cumsum;
                return (*bufs[i])[relidx];
            }
            cumsum += bufs[i]->size();
        }
        return 0;
    }
};

struct counts_merge_helper {
    counts_merge_helper(std::vector<counts_writer>& cws)
        : writers(cws)
    {
    }
    ~counts_merge_helper()
    {
        for (auto& w : writers) {
            w.delete_temp_files();
        }
    }

    sdsl::bit_vector get_bv(size_t n)
    {
        sdsl::bit_vector bv(n);
        for (auto& w : writers) {
            for (const auto& id : w.ids) {
                bv[id] = 1;
            }
        }
        return bv;
    }

    dummy_container get_counts_f1()
    {
        std::vector<sdsl::int_vector_buffer<32>*> bufs;
        for (auto& w : writers) {
            bufs.push_back(&w.f1);
        }
        return dummy_container(bufs);
    }

    dummy_container get_counts_f2()
    {
        std::vector<sdsl::int_vector_buffer<32>*> bufs;
        for (auto& w : writers) {
            bufs.push_back(&w.f2);
        }
        return dummy_container(bufs);
    }

    dummy_container get_counts_fb()
    {
        std::vector<sdsl::int_vector_buffer<32>*> bufs;
        for (auto& w : writers) {
            bufs.push_back(&w.fb);
        }
        return dummy_container(bufs);
    }

    dummy_container get_counts_b()
    {
        std::vector<sdsl::int_vector_buffer<32>*> bufs;
        for (auto& w : writers) {
            bufs.push_back(&w.b);
        }
        return dummy_container(bufs);
    }

    dummy_container get_counts_f1prime()
    {
        std::vector<sdsl::int_vector_buffer<32>*> bufs;
        for (auto& w : writers) {
            bufs.push_back(&w.f1p);
        }
        return dummy_container(bufs);
    }

    dummy_container get_counts_f2prime()
    {
        std::vector<sdsl::int_vector_buffer<32>*> bufs;
        for (auto& w : writers) {
            bufs.push_back(&w.f2p);
        }
        return dummy_container(bufs);
    }

private:
    std::vector<counts_writer>& writers;
};
}