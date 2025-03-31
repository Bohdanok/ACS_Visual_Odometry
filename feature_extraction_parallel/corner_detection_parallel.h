//
// Created by julfy1 on 3/24/25.
//
#pragma once

#ifndef CORNER_DETECTION
#define CORNER_DETECTION

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// struct point {
//     int x, y;
// };

constexpr size_t NUMBER_OF_THREADS = 3;
constexpr size_t BLOCK_SIZE = 20;
constexpr double RESPONSE_THRESHOLD = 1e5;

struct bound2d {
    int start{}, end{};
};

struct interval {
    bound2d cols, rows;
};

enum WORKER_FUNCTION {
    COMPUTE_CORNERS = 0,
    COMPUTE_LOCAL_MINS = 1,
    COMPUTE_DESCRIPTORS = 2
};



class CornerDetection {
public:
    static cv::Mat custom_bgr2gray(cv::Mat& picture);
    // static void compare_images(const cv::Mat& image_my, const cv::Mat& image_their, const std::string win_name);
    // static std::vector<cv::Mat> direction_gradients(cv::Mat& picture, const int& n_rows, const int& n_cols);
    // static std::vector<std::vector<double>> shitomasi_corner_detection(cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, const int& n_rows, const int& n_cols, const double& k);
    // static void draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name);
    static std::vector<cv::KeyPoint> non_maximum_suppression(std::vector<std::vector<double>> R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);

    static std::vector<cv::Mat> direction_gradients_worker(const cv::Mat& picture, const interval& interval);
    static std::vector<std::vector<double>> shitomasi_corner_detection_worker(const cv::Mat& Jx, const cv::Mat& Jy, const cv::Mat& Jxy, const interval& interval, const double& k);
    // static std::vector<cv::KeyPoint> non_maximum_suppression_worker(std::vector<std::vector<double>> R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);


};

#endif // CORNER_DETECTION