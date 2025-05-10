//
// Created by julfy on 13.04.25.
//

#ifndef TEST_OPENCL_H
#define TEST_OPENCL_H
#include <CL/cl2.hpp>

constexpr double RESPONSE_THRESHOLD = 25000;

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

unsigned int get_best_opencl_platform();

cl::Program create_platform(const std::string &filename);

#endif //TEST_OPENCL_H
