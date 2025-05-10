//
// Created by julfy on 20.04.25.
//

#ifndef FEATURE_MATCHING_PARALLEL_H
#define FEATURE_MATCHING_PARALLEL_H

#include <vector>
#include <utility>
#include <cstdint>
#include <optional>
#include <chrono>
#include <atomic>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

// Forward declaration
class thread_pool;

constexpr int BINARY_DESCRIPTOR_SIZE = 32;
constexpr double MATCH_THRESHOLD = 0.5;


inline std::chrono::high_resolution_clock::time_point get_current_time_fenced();

inline int hammingDistance(const uint8_t* d1, const uint8_t* d2);

std::vector<std::pair<int, int>> matchCustomBinaryDescriptorsThreadPool(
    const std::vector<std::vector<uint8_t>>& desc1,
    const std::vector<std::vector<uint8_t>>& desc2,
    thread_pool& pool,
    int numThreads,
    float ratioThreshold
);

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const;
};

std::vector<std::pair<int, int>> findCommonMatches(
    const std::vector<std::pair<int, int>>& customMatches,
    const std::vector<cv::DMatch>& cvMatches
);

std::vector<cv::DMatch> convertToDMatch(const std::vector<std::pair<int, int>>& matches);

std::vector<cv::DMatch> applyRANSAC(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2
);

#endif // FEATURE_MATCHING_PARALLEL_H
