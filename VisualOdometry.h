//
// Created by julfy on 20.04.25.
//

#ifndef MAIN_H
#define MAIN_H

#include "feature_extraction_parallel/feature_extraction_parallel.h"
#include "feature_matching_parallel/feature_matching_parallel.h"


class VisualOdometry {
private:
    thread_pool VO_pool;
    const size_t number_of_threads;
public:
    explicit VisualOdometry(const size_t number_of_threads);
    std::vector<std::vector<uint8_t>> compute_descriptor(cv::Mat& image);
    std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> compute_descriptor_with_key_points(cv::Mat& image);
    std::vector<std::pair<int, int>> match_descriptors(const std::vector<std::vector<uint8_t>>& desc1, const std::vector<std::vector<uint8_t>>& desc2);
};



#endif //MAIN_H
