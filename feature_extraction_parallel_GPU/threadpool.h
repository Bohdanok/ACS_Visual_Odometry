//
// Created by julfy1 on 3/31/25.
//

#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <future>
#include "join_threads.h"
#include "ts_queue.h"

class thread_pool {
    std::atomic_bool done;
    TS_QUEUE<std::function<void()>> work_queue;
    std::vector<std::thread> threads;
    join_threads joiner;

    void worker_thread() {
        while (!done) {
            // Block until a task is available
            std::function<void()> task = work_queue.dequeue();
            if (task) task();  // Execute the task
        }
    }

public:
    thread_pool(unsigned int thread_count) : done(false), joiner(threads) {
        try {
            for (unsigned i = 0; i < thread_count; ++i) {
                threads.emplace_back(&thread_pool::worker_thread, this);
            }
        } catch (...) {
            done = true;
            throw;
        }
    }

    ~thread_pool() {
        done = true;
        // Push dummy tasks to unblock all threads
        for (size_t i = 0; i < threads.size(); ++i) {
            work_queue.enqueue([]() {});
        }
        // join all threads
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    }


    template<typename FunctionType>
    auto submit(FunctionType f) -> std::future<decltype(f())> {
        using result_type = decltype(f());
        auto task = std::make_shared<std::packaged_task<result_type()>>(std::move(f));
        std::future<result_type> res = task->get_future();
        work_queue.enqueue([task]() { (*task)(); });
        return res;
    }
};

#endif //THREADPOOL_H
