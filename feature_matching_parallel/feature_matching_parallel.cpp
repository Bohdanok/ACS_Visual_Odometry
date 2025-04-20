//
// Created by julfy on 20.04.25.
//

#include "feature_matching_parallel.h"


#include "feature_extraction_parallel/threadpool.h"
#include <iostream>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <bitset>
#include <thread>
#include <future>
#include <opencv2/opencv.hpp>
#include <optional>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#ifdef PARALLEL_IMPLEMENTATION
    #include "feature_extraction_parallel/feature_extraction_parallel.h"
#else
    #include "feature_extraction/test_feature_extraction.h"
#endif


// #define VISUALIZATION
#define TIME_MEASUREMENT


constexpr int BINARY_DESCRIPTOR_SIZE = 32;
constexpr double MATCH_THRESHOLD = 0.5;

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

int hammingDistance(const uint8_t* d1, const uint8_t* d2, int length) {
    int distance = 0;
    int i = 0;

    for (; i + 4 <= length; i += 4) {
        uint32_t v1, v2;
        std::memcpy(&v1, d1 + i, sizeof(uint32_t));
        std::memcpy(&v2, d2 + i, sizeof(uint32_t));
        distance += __builtin_popcount(v1 ^ v2);
    }

    for (; i < length; ++i) {
        distance += __builtin_popcount(d1[i] ^ d2[i]);
    }

    return distance;
}

std::vector<std::pair<int, int>> matchCustomBinaryDescriptorsParallel(
    const std::vector<std::vector<uint8_t>>& desc1,
    const std::vector<std::vector<uint8_t>>& desc2,
    float ratioThreshold = 0.75f,
    int numThreads = std::thread::hardware_concurrency())
{
    std::vector<std::pair<int, int>> allMatches;
    if (desc1.empty() || desc2.empty()) return allMatches;

    int descriptorLength = desc1[0].size();
    size_t total = desc1.size();
    // const int granularity = 128;
    // size_t chunkSize = std::max(desc1.size() / (numThreads * granularity), size_t(1));

    size_t chunkSize = (total + numThreads - 1) / numThreads;

    std::vector<std::future<std::vector<std::pair<int, int>>>> futures;

    for (int t = 0; t < numThreads; ++t) {
        size_t startIdx = t * chunkSize;
        size_t endIdx = std::min(startIdx + chunkSize, total);

        futures.emplace_back(std::async(std::launch::async, [&, startIdx, endIdx]() {
            std::vector<std::pair<int, int>> localMatches;
            for (size_t i = startIdx; i < endIdx; ++i) {
                int bestIdx = -1, secondBestIdx = -1;
                int bestDist = std::numeric_limits<int>::max();
                int secondBestDist = std::numeric_limits<int>::max();

                for (size_t j = 0; j < desc2.size(); ++j) {
                    int dist = hammingDistance(desc1[i].data(), desc2[j].data(), descriptorLength);
                    if (dist < bestDist) {
                        secondBestDist = bestDist;
                        secondBestIdx = bestIdx;
                        bestDist = dist;
                        bestIdx = j;
                    } else if (dist < secondBestDist) {
                        secondBestDist = dist;
                        secondBestIdx = j;
                    }
                }

                if (bestIdx != -1 && secondBestIdx != -1 &&
                    bestDist < ratioThreshold * secondBestDist) {
                    localMatches.emplace_back(i, bestIdx);
                }
            }
            return localMatches;
        }));
    }

    for (auto& f : futures) {
        auto result = f.get();
        allMatches.insert(allMatches.end(), result.begin(), result.end());
    }

    return allMatches;
}

std::vector<std::pair<int, int>> matchCustomBinaryDescriptorsThreadPool(
    const std::vector<std::vector<uint8_t>>& desc1,
    const std::vector<std::vector<uint8_t>>& desc2,
    thread_pool& pool,
    int numThreads,
    float ratioThreshold = 0.75f)
{
    std::vector<std::pair<int, int>> allMatches;
    if (desc1.empty() || desc2.empty()) return allMatches;

    const int descriptorLength = desc1[0].size();
    const size_t total = desc1.size();
    const size_t chunkSize = (total + numThreads - 1) / numThreads;

    std::vector<std::future<std::vector<std::pair<int, int>>>> futures;

    for (int t = 0; t < numThreads; ++t) {
        size_t startIdx = t * chunkSize;
        size_t endIdx = std::min(startIdx + chunkSize, total);
        if (startIdx >= total) break;

        futures.emplace_back(pool.submit([&, startIdx, endIdx]() {
            std::vector<std::pair<int, int>> localMatches;
            localMatches.reserve(chunkSize);

            for (size_t i = startIdx; i < endIdx; ++i) {
                int bestIdx = -1, secondBestIdx = -1;
                int bestDist = std::numeric_limits<int>::max();
                int secondBestDist = std::numeric_limits<int>::max();

                for (size_t j = 0; j < desc2.size(); ++j) {
                    int dist = hammingDistance(desc1[i].data(), desc2[j].data(), descriptorLength);
                    if (dist < bestDist) {
                        secondBestDist = bestDist;
                        secondBestIdx = bestIdx;
                        bestDist = dist;
                        bestIdx = j;
                    } else if (dist < secondBestDist) {
                        secondBestDist = dist;
                        secondBestIdx = j;
                    }
                }

                if (bestIdx != -1 && secondBestIdx != -1 &&
                    bestDist < ratioThreshold * secondBestDist) {
                    localMatches.emplace_back(i, bestIdx);
                }
            }

            return localMatches;
        }));
    }

    for (auto& f : futures) {
        std::vector<std::pair<int, int>> result = std::move(f.get());
        allMatches.insert(allMatches.end(),
                          std::make_move_iterator(result.begin()),
                          std::make_move_iterator(result.end()));
    }

    return allMatches;
}


struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};


std::vector<std::pair<int, int>> findCommonMatches(
    const std::vector<std::pair<int, int>>& customMatches,
    const std::vector<cv::DMatch>& cvMatches)
{
    std::unordered_set<std::pair<int, int>, PairHash> cvMatchSet;
    for (const auto& cvm : cvMatches) {
        cvMatchSet.insert({cvm.queryIdx, cvm.trainIdx});
    }
    std::vector<std::pair<int, int>> commonMatches;
    for (const auto& custom : customMatches) {
        if (cvMatchSet.find(custom) != cvMatchSet.end()) {
            commonMatches.push_back(custom);
        }
    }
    return commonMatches;
}

std::vector<cv::DMatch> convertToDMatch(const std::vector<std::pair<int, int>>& matches) {
    std::vector<cv::DMatch> dMatches;
    for (const auto& match : matches) {
        dMatches.emplace_back(cv::DMatch(match.first, match.second, 0));
    }
    return dMatches;
}

std::vector<cv::DMatch> applyRANSAC(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2)
{
    if (matches.size() < 4) return matches;

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(keypoints1[match.queryIdx].pt);
        pts2.push_back(keypoints2[match.trainIdx].pt);
    }

    std::vector<uchar> inliersMask;
    cv::Mat homography = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, inliersMask);

    std::vector<cv::DMatch> filteredMatches;
    for (size_t i = 0; i < matches.size(); i++) {
        if (inliersMask[i]) {
            filteredMatches.push_back(matches[i]);
        }
    }
    return filteredMatches;
}

std::vector<std::pair<int, int>> matchCustomBinaryDescriptorsThreadPool_v2(
    const std::vector<std::vector<uint8_t>>& desc1,
    const std::vector<std::vector<uint8_t>>& desc2,
    thread_pool& pool,
    float ratioThreshold = 0.75f)
{
    const int descriptorLength = desc1[0].size();
    std::vector<std::future<std::optional<std::pair<int, int>>>> futures;

    for (size_t i = 0; i < desc1.size(); ++i) {
        futures.emplace_back(pool.submit([=, &desc1, &desc2]() -> std::optional<std::pair<int, int>> {
            int bestIdx = -1, secondBestIdx = -1;
            int bestDist = std::numeric_limits<int>::max();
            int secondBestDist = std::numeric_limits<int>::max();

            for (size_t j = 0; j < desc2.size(); ++j) {
                int dist = hammingDistance(desc1[i].data(), desc2[j].data(), descriptorLength);
                if (dist < bestDist) {
                    secondBestDist = bestDist;
                    secondBestIdx = bestIdx;
                    bestDist = dist;
                    bestIdx = j;
                } else if (dist < secondBestDist) {
                    secondBestDist = dist;
                    secondBestIdx = j;
                }
            }

            if (bestIdx != -1 && secondBestIdx != -1 &&
                bestDist < ratioThreshold * secondBestDist) {
                return std::make_pair(i, bestIdx);
            }
            return std::nullopt;
        }));
    }

    std::vector<std::pair<int, int>> allMatches;
    allMatches.reserve(desc1.size());

    for (auto& f : futures) {
        if (auto opt = f.get()) {
            allMatches.push_back(*opt);
        }
    }

    return allMatches;
}
