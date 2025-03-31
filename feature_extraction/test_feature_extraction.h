//
// Created by julfy1 on 3/24/25.
//
#pragma once

#include "corner_detection.h"
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#ifndef TEST_FEATURE_EXTRACTION_H
#define TEST_FEATURE_EXTRACTION_H



void compare_images(const cv::Mat& image_my, const cv::Mat& image_their, const std::string win_name);
void draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name);
void print_descriptor(const std::vector<std::vector<uint8_t>>& descriptor);
std::vector<std::vector<uint8_t>> descriptor_for_s_pop(const std::string& filename);
std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> descriptor_with_points(const std::string& filename);

#endif //TEST_FEATURE_EXTRACTION_H
