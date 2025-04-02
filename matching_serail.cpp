#include <iostream>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <bitset>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "feature_extraction/test_feature_extraction.h"

#define VISUALIZATION

constexpr int BINARY_DESCRIPTOR_SIZE = 32;
constexpr double MATCH_THRESHOLD = 0.5;

inline std::chrono::high_resolution_clock::time_point get_current_time_fenced() {
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

std::vector<std::pair<int, int>> matchCustomBinaryDescriptorsSerial(
    const std::vector<std::vector<uint8_t>>& desc1,
    const std::vector<std::vector<uint8_t>>& desc2,
    float ratioThreshold = 0.75f)
{
    std::vector<std::pair<int, int>> matches;
    if (desc1.empty() || desc2.empty()) return matches;

    int descriptorLength = desc1[0].size();

    for (size_t i = 0; i < desc1.size(); ++i) {
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
            matches.emplace_back(i, bestIdx);
        }
    }

    return matches;
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./matching <image1> <image2>" << std::endl;
        return -1;
    }

    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error reading images." << std::endl;
        return -1;
    }

    auto start_feature_extraction = get_current_time_fenced();
    auto descs1 = descriptor_with_points(argv[1]);
    auto descs2 = descriptor_with_points(argv[2]);
    auto end_feature_extraction = get_current_time_fenced();

    std::cout << "Number of keypoints (img1): " << std::get<1>(descs1).size() << std::endl;
    std::cout << "Number of keypoints (img2): " << std::get<1>(descs2).size() << std::endl;

    std::cout << "Time for feature extraction: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_feature_extraction - start_feature_extraction).count()
              << " ms" << std::endl;

    std::vector<std::pair<int, int>> customMatches;
    auto start = get_current_time_fenced();
    customMatches = matchCustomBinaryDescriptorsSerial(std::get<0>(descs1), std::get<0>(descs2));
    auto end = get_current_time_fenced();

    std::cout << "Time for matching: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    std::cout << "Custom binary matches: " << customMatches.size() << std::endl;

#ifdef VISUALIZATION
    cv::Mat binaryMatchesImg;
    cv::drawMatches(img1, std::get<1>(descs1), img2, std::get<1>(descs2), convertToDMatch(customMatches), binaryMatchesImg);
    cv::imshow("BRISK Matches", binaryMatchesImg);
    cv::waitKey(0);
#endif

    return 0;
}
