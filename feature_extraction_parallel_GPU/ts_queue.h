//
// Created by julfy1 on 3/31/25.
//

#ifndef TS_QUEUE_H
#define TS_QUEUE_H
#include <condition_variable>
#include <deque>

template<typename T>
class TS_QUEUE {

    std::deque<T> data;
    std::mutex m;
    std::condition_variable cv;
    size_t length = 0;

public:

    void enqueue (const T& d) {
        {
            std::lock_guard guard(m);
            // length++;
            data.push_back(d);

        }
        cv.notify_one();
    }

    T dequeue() {
        std::unique_lock<std::mutex> u_lock(m);
        while (data.empty()) {
            cv.wait(u_lock);
        }
        auto t = data.front();
        data.pop_front();
        return t;
    }

    bool empty() {
        std::lock_guard guard(m);
        return data.empty();
    }

    // void print_queue() {
    //     while (!this->empty()) {
    //         auto q_entry = this->get();
    //         std::cout << "Queue Entry: Bound Start: " << q_entry.bound_start.x << ":" <<q_entry.bound_start.y << ", Steps Start: (" << q_entry.steps_start.x << ", " << q_entry.steps_start.y << "), Delta: (" << q_entry.delta.x << ", " << q_entry.delta.y << "), Points Per Task: " << q_entry.points_per_task << ", Steps Size: (" << q_entry.steps_size.x << ", " << q_entry.steps_size.y << "), Optimized: " << (q_entry.optimized ? "true" : "false") << "\n";
    //
    //     }
    // }


    size_t get_length() {
        std::lock_guard guard(m);
        return data.size();
    }


};

#endif //TS_QUEUE_H
