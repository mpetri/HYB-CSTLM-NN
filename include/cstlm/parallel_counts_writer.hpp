#pragma once

// containers
#include <vector>
#include <queue>
// threading
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
// utility wrappers
#include <memory>
#include <functional>
// exceptions
#include <stdexcept>
#include <unordered_map>

#include "logging.hpp"
#include "utils.hpp"

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

template <class t_func>
class parallel_counts_writer {
public:
    // the constructor just launches some amount of workers
    parallel_counts_writer(t_func write_func, std::vector<uint64_t>& nds)
        : write_item_func(write_func)
        , stop(false)
        , next_id_to_write(0)
        , nodes(nds)
    {

        worker_thread = std::thread(
            [this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return (this->results.count(this->nodes[this->next_id_to_write]) != 0); });

                        while (this->next_id_to_write != this->nodes.size() && this->results.count(nodes[next_id_to_write]) != 0) {
                            auto itr = this->results.find(this->nodes[this->next_id_to_write]);
                            const auto& items = itr->second;

                            write_item_func(items);

                            this->results.erase(itr);
                            next_id_to_write++;
                        }
                        if (this->next_id_to_write == this->nodes.size())
                            return;
                    }
                }
            });
    }
    // deleted copy&move ctors&assignments
    parallel_counts_writer(const parallel_counts_writer&) = delete;
    parallel_counts_writer& operator=(const parallel_counts_writer&) = delete;
    parallel_counts_writer(parallel_counts_writer&&) = delete;
    parallel_counts_writer& operator=(parallel_counts_writer&&) = delete;
    // add new work item to the pool
    void write_results(uint64_t node_id, std::vector<computed_context_result>& results)
    {
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->results.emplace(node_id, std::move(results));
        }
        this->condition.notify_one();
    }

    // the destructor joins all threads
    virtual ~parallel_counts_writer()
    {
        this->stop = true;
        this->condition.notify_all();
        worker_thread.join();
        LOG(INFO) << "Processed " << next_id_to_write << " / " << nodes.size() << " subtrees.";
    }

private:
    // need to keep track of threads so we can join them
    std::thread worker_thread;
    t_func write_item_func;
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    // workers finalization flag
    std::atomic_bool stop;
    std::atomic_uint_least64_t next_id_to_write;
    std::vector<uint64_t> nodes;
    std::unordered_map<uint64_t, std::vector<computed_context_result> > results;
};
}