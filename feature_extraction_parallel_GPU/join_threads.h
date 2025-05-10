//
// Created by julfy1 on 3/31/25.
//

#ifndef JOIN_THREADS_H
#define JOIN_THREADS_H

#include <vector>
#include <thread>

class join_threads {
    std::vector<std::thread>& threads;
public:
    explicit join_threads(std::vector<std::thread>& threads_) : threads(threads_) {}

    ~join_threads() {
        for (auto& t : threads) {
            if (t.joinable())
                t.join();
        }
    }
};

#endif //JOIN_THREADS_H
