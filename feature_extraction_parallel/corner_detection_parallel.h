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

// struct point {
//     int x, y;
// };

constexpr size_t NUMBER_OF_THREADS = 3;
constexpr int BLOCK_SIZE = 20;
constexpr double RESPONSE_THRESHOLD = 25000;

struct bound2d {
    int start{}, end{};
};

struct interval {
    bound2d cols, rows;
};

enum WORKER_FUNCTION {
    COMPUTE_CORNERS,
    COMPUTE_LOCAL_MINS,
    COMPUTE_DESCRIPTORS
};

using work_result = std::variant<bool, std::vector<cv::KeyPoint>, std::vector<std::vector<uint8_t>>>;

class CornerDetectionParallel {
public:
    static cv::Mat custom_bgr2gray(cv::Mat& picture);
    static std::vector<cv::KeyPoint> non_maximum_suppression(const std::vector<std::vector<double>> &R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);

    static void direction_gradients_worker(const cv::Mat& picture, const interval& interval, cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy);
    static void shitomasi_corner_detection_worker(const cv::Mat& Jx, const cv::Mat& Jy, const cv::Mat& Jxy, const interval& interval, const double& k, std::vector<std::vector<double>>& R_array);
    // static std::vector<cv::KeyPoint> non_maximum_suppression_worker(std::vector<std::vector<double>> R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);


};

#endif // CORNER_DETECTION_PARALLEL