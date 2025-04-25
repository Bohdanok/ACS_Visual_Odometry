//
// Created by julfy1 on 3/24/25.
//
#pragma once
#ifndef CORNER_DETECTION_PARALLEL
#define CORNER_DETECTION_PARALLEL

#include <opencv2/core.hpp>
#include <string>
#include <variant>
#include <vector>
#include <CL/cl2.hpp>

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

constexpr float RESPONSE_THRESHOLD = 25000;

struct GPU_settings {
    cl::Program program;
    cl::Device device;
    cl::Context context;
};



using work_result = std::variant<bool, std::vector<cv::KeyPoint>, std::vector<std::vector<uint8_t>>>;

class CornerDetectionParallel_GPU {
public:
    static cv::Mat custom_bgr2gray(cv::Mat& picture);
    // static std::vector<cv::KeyPoint> non_maximum_suppression_worker(const std::vector<std::vector<double>> &R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);

    static void shitomasi_corner_detection(const GPU_settings& gpu_settings, const cv::Mat& my_blurred_gray, std::vector<std::vector<float>>& R_score);
    static std::vector<cv::KeyPoint> non_maximum_suppression(const std::vector<std::vector<float>> &R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);


};

#endif // CORNER_DETECTION_PARALLEL