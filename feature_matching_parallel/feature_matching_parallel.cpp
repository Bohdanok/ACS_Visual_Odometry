#include "feature_matching_parallel.h"


#include "../feature_extraction_parallel/threadpool.h"
#include "../feature_extraction_parallel/feature_extraction_parallel.h"

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
// #include <arm_neon.h>



// #ifdef PARALLEL_IMPLEMENTATION
//     #include "feature_extraction_parallel/feature_extraction_parallel.h"
// #else
//     #include "feature_extraction/test_feature_extraction.h"
// #endif
//

// #define VISUALIZATION

std::size_t PairHash::operator()(const std::pair<int, int>& p) const {
    return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
}

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

inline int hammingDistance(const uint8_t* d1, const uint8_t* d2) {
    const uint64_t* a = reinterpret_cast<const uint64_t*>(d1);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(d2);

    int dist = 0;
    for (int i = 0; i < 64; ++i) {
        dist += __builtin_popcountll(a[i] ^ b[i]);
    }
    return dist;
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
        constexpr int INF = std::numeric_limits<int>::max();

        futures.emplace_back(pool.submit([&, startIdx, endIdx]() {
            std::vector<std::pair<int, int>> localMatches;
            localMatches.reserve(chunkSize);

            for (size_t i = startIdx; i < endIdx; ++i) {
                int bestIdx = -1, secondBestIdx = -1;
                int bestDist = INF;
                int secondBestDist = INF;

                for (size_t j = 0; j < desc2.size(); ++j) {
                    int dist = hammingDistance(desc1[i].data(), desc2[j].data());
                    if (dist >= secondBestDist) continue;
                    if (dist < bestDist) {
                        secondBestDist = bestDist;
                        secondBestIdx = bestIdx;
                        bestDist = dist;
                        bestIdx = j;
                    }
                    else if (dist < secondBestDist) {
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
