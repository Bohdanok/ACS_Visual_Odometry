//
// Created by julfy1 on 3/24/25.
//
#pragma once
#ifndef CORNER_DETECTION
#define CORNER_DETECTION

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct point {
    int x, y;
};

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
    static std::vector<point> non_maximum_suppression(std::vector<std::vector<double>> R_values, const int& n_rows, const int& n_cols, const int& k, const int& N);


    // static auto prepare_the_surroundings(const cv::Mat& blurred_gray_picture, const std::vector<int>& key_point, const int& n_cols, const int& n_rows);
    // static double compute_orientation(point point, cv::Mat& image, const int& n_cols, const int& n_rows);
    // static auto add_transponed_vector(std::vector<std::vector<int>>& array, const std::vector<double>& add_vector, const size_t index, const size_t num_of_keypoints);
    // static auto FREAK_feature_description(const std::vector<std::tuple<int, int, double>>& key_points, cv::Mat blurred_gray_picture, const int& n_cols, const int& n_rows, const double corr_threshold);
    // static auto prepare_and_test(const std::string& filename, const std::string& cur_path, const std::string& win_name, const bool draw = false);
};

#endif // CORNER_DETECTION