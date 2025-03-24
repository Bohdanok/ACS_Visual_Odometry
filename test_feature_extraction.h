//
// Created by julfy1 on 3/24/25.
//
#pragma once

#include "corner_detection.h"

#ifndef TEST_FEATURE_EXTRACTION_H
#define TEST_FEATURE_EXTRACTION_H



void compare_images(const cv::Mat& image_my, const cv::Mat& image_their, const std::string win_name);
void draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name);


#endif //TEST_FEATURE_EXTRACTION_H
