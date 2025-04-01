//
// Created by julfy1 on 4/1/25.
//
#pragma once

#ifndef FEATURE_EXTRACTION_PARALLEL_H
#define FEATURE_EXTRACTION_PARALLEL_H
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "threadpool.h"
#include "corner_detection_parallel.h"
#include "FREAK_feature_descriptor_parallel.h"

// #define VISUALIZATION

void print_descriptor(const std::vector<std::vector<uint8_t>>& descriptor);
void response_worker(const cv::Mat& blurred_gray, const interval& interval, cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, std::vector<std::vector<double>>& R_array);
std::vector<std::vector<uint8_t>> feature_extraction_manager(cv::Mat& image, thread_pool& pool);
std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> feature_extraction_manager_with_points(cv::Mat& image, thread_pool& pool);


#endif //FEATURE_EXTRACTION_PARALLEL_H
