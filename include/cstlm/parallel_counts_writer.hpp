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

struct computed_context_result {
    uint64_t id;
    uint32_t f1;
    uint32_t f2;
    uint32_t fb;
    uint32_t b;
    uint32_t f1prime;
    uint32_t f2prime;
    bool operator>(const computed_context_result& other) const
    {
        return id > other.id;
    }
    bool operator<(const computed_context_result& other) const
    {
        return id < other.id;
    }
};

template <class t_func>
class parallel_counts_writer {
public:
    // the constructor just launches some amount of workers
    parallel_counts_writer(t_func write_func)
        : write_item_func(write_func)
        , stop(false)
        , next_id_to_write(0)
    {

        worker_thread = std::thread(
            [this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || (!this->items.empty() && this->next_id_to_write == this->items.top().id); });

                        while (!this->items.empty() && (next_id_to_write == this->items.top().id)) {
                            const auto& item = this->items.top();

                            write_item_func(item);

                            this->items.pop();
                            next_id_to_write++;
                        }
                        if (this->stop)
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
    void enqueue(computed_context_result&& new_item)
    {
        size_t items_in_queue = 0;
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->items.emplace(new_item);
            items_in_queue = this->items.size();
        }
        if (items_in_queue > 2048)
            this->condition.notify_one();
    }
    template <class t_vec>
    void enqueue_bulk(t_vec& new_items, size_t num_items)
    {
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            for (size_t i = 0; i < num_items; i++)
                this->items.emplace(new_items[i]);
        }
        this->condition.notify_one();
    }

    size_t queue_size() const
    {
        return this->items.size();
    }
    // the destructor joins all threads
    virtual ~parallel_counts_writer()
    {
        this->stop = true;
        this->condition.notify_all();
        worker_thread.join();
    }

private:
    // need to keep track of threads so we can join them
    std::thread worker_thread;
    // the task queue
    std::priority_queue<computed_context_result, std::vector<computed_context_result>, std::greater<computed_context_result> > items;
    t_func write_item_func;
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    // workers finalization flag
    std::atomic_bool stop;
    std::atomic_uint_least64_t next_id_to_write;
};