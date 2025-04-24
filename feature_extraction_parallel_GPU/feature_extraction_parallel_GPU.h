//
// Created by julfy1 on 4/1/25.
//
#pragma once

#ifndef FEATURE_EXTRACTION_PARALLEL_H
#define FEATURE_EXTRACTION_PARALLEL_H

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "threadpool.h"
#include "corner_detection_parallel_GPU.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <CL/cl2.hpp>
#include <opencv2/opencv.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS


#ifndef CORNER_DETECTION_PARALLEL
struct GPU_settings {
    cl::Program program;
    cl::Device device;
    cl::Context context;
};
#endif

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}


bool is_number(const std::string& s);
void draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name);
void print_descriptor(const std::vector<std::vector<uint8_t>>& descriptor);


cl::Program create_platform_from_binary(const std::string &binary_filename);


std::vector<std::vector<uint8_t>> feature_extraction_manager(const cv::Mat& image);



#endif //FEATURE_EXTRACTION_PARALLEL_H
