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

class CornerDetection {
public:
    static cv::Mat custom_bgr2gray(cv::Mat& picture);
    static cv::Mat test_their_sobel(const std::string& filename, const std::string& cur_path);
    // static void compare_images(const cv::Mat& image_my, const cv::Mat& image_their, const std::string win_name);
    static std::vector<cv::Mat> direction_gradients(cv::Mat& picture, const int& n_rows, const int& n_cols);
    static cv::Mat sobel_filter(cv::Mat& Jx, cv::Mat& Jy, const int& n_rows, const int& n_cols);
    static std::vector<std::vector<double>> harris_corner_detection(cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, const int& n_rows, const int& n_cols, const double& k);
    static std::vector<std::vector<double>> shitomasi_corner_detection(cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, const int& n_rows, const int& n_cols, const double& k);
    // static void draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name);
    static std::vector<cv::KeyPoint> non_maximum_suppression(std::vector<std::vector<double>> R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);

};

#endif // CORNER_DETECTION