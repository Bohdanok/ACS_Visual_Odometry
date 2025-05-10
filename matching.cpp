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

inline int hammingDistance(const uint8_t* d1, const uint8_t* d2) {
    const uint64_t* a = reinterpret_cast<const uint64_t*>(d1);
    const uint64_t* b = reinterpret_cast<const uint64_t*>(d2);

    return __builtin_popcountll(a[0] ^ b[0]) +
           __builtin_popcountll(a[1] ^ b[1]) +
           __builtin_popcountll(a[2] ^ b[2]) +
           __builtin_popcountll(a[3] ^ b[3]);
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
                    int dist = hammingDistance(desc1[i].data(), desc2[j].data());
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

cv::Mat convertToCvDescriptorsMatrix(const std::vector<std::vector<uint8_t>>& customDescriptors) {
    // Ensure descriptors are in the correct format for BFMatcher: a cv::Mat with descriptors as rows
    int numDescriptors = customDescriptors.size();
    if (numDescriptors == 0) return cv::Mat(); // Empty case
    
    int descriptorLength = customDescriptors[0].size();
    
    cv::Mat descriptors(numDescriptors, descriptorLength, CV_8UC1);
    
    for (int i = 0; i < numDescriptors; i++) {
        std::memcpy(descriptors.ptr(i), customDescriptors[i].data(), descriptorLength);
    }
    
    return descriptors;
}



int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./matching <image1> <image2>" << std::endl;
        return -1;
    }

    int NUMBER_OF_THREADS = std::stoi(argv[3]);
    thread_pool pool1(NUMBER_OF_THREADS);

    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error reading images." << std::endl;
        return -1;
    }
#ifdef PARALLEL_IMPLEMENTATION
    if (!is_number(argv[3]))
    {
        std::cerr << "The number of threads has to be a positive number!" << std::endl;
        return 69;
    }

    if (!is_number(argv[4]))
    {
        std::cerr << "The block size has to be a positive number!" << std::endl;
        return -69;
    }

    BLOCK_SIZE = std::stoi(argv[4]);
#endif

    auto start_feature_extraction = get_current_time_fenced();

#ifdef PARALLEL_IMPLEMENTATION
    cv::Mat image1 = cv::imread(argv[1]);
    cv::Mat image2 = cv::imread(argv[2]);

    auto descs1 = feature_extraction_manager_with_points(image1, pool1);
    auto descs2 = feature_extraction_manager_with_points(image2, pool1);

#else
    auto descs1 = descriptor_with_points(argv[1]);
    auto descs2 = descriptor_with_points(argv[2]);
#endif

    auto end_feature_extraction = get_current_time_fenced();

    std::cout << "Number of keypoints: " << std::get<1>(descs1).size() << std::endl;
    std::cout << "Number of keypoints: " << std::get<1>(descs2).size() << std::endl;

    std::cout << "Time for feature extraction: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_feature_extraction - start_feature_extraction).count()
              << " ms" << std::endl;

    std::vector<std::pair<int, int>> customMatches;
    auto start = get_current_time_fenced();
    customMatches = matchCustomBinaryDescriptorsThreadPool(
    	std::get<0>(descs1),
    	std::get<0>(descs2),
    	pool1,
    	NUMBER_OF_THREADS
	);
    auto end = get_current_time_fenced();

    std::cout << "Time for matching: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    std::cout << "Custom binary matches: " << customMatches.size() << std::endl;

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> openCVMatches;
    cv::Mat cvDescriptors1 = convertToCvDescriptorsMatrix(std::get<0>(descs1));
    cv::Mat cvDescriptors2 = convertToCvDescriptorsMatrix(std::get<0>(descs2));

    auto startMatching = get_current_time_fenced();
    matcher.match(cvDescriptors1, cvDescriptors2, openCVMatches);
    

    auto endMatching = get_current_time_fenced();

    std::cout << "Time for matching with OpenCV: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(endMatching - startMatching).count()
            << " ms" << std::endl;



#ifdef VISUALIZATION
    cv::Mat binaryMatchesImg;
    cv::drawMatches(img1, std::get<1>(descs1), img2, std::get<1>(descs2), convertToDMatch(customMatches), binaryMatchesImg);
    cv::imshow("FREAK Matches", binaryMatchesImg);
    cv::waitKey(0);
#endif

    return 0;
}
