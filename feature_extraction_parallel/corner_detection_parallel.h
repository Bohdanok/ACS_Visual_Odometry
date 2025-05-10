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


inline size_t NUMBER_OF_THREADS = 4;
inline int BLOCK_SIZE = 100;
constexpr float RESPONSE_THRESHOLD = 25000;

struct bound2d {
    int start{}, end{};
};

struct interval {
    bound2d cols, rows;
};

struct Candidate {
    double score;
    int i;
    int j;

    bool operator<(const Candidate& other) const {
        return score > other.score;
    }
};

using work_result = std::variant<bool, std::vector<cv::KeyPoint>, std::vector<std::vector<uint8_t>>>;

class CornerDetectionParallel {
public:
    static cv::Mat custom_bgr2gray(cv::Mat& picture);

    static void direction_gradients_worker(const cv::Mat& picture, const interval& interval, cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy);
    static void shitomasi_corner_detection_worker(const cv::Mat& Jx, const cv::Mat& Jy, const cv::Mat& Jxy, const interval& interval, const float& k, std::vector<std::vector<float>>& R_array);
    static std::vector<cv::KeyPoint> non_maximum_suppression(const std::vector<std::vector<float>> &R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);


};

#endif // CORNER_DETECTION_PARALLEL