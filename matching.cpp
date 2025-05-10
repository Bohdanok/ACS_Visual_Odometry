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

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}


#elif  GPU_IMPLEMENTATION
#include "feature_extraction_parallel_GPU/feature_extraction_parallel_GPU.h"
#else
    #include "feature_extraction/test_feature_extraction.h"
#endif


// #define VISUALIZATION

constexpr int BINARY_DESCRIPTOR_SIZE = 32;
constexpr double MATCH_THRESHOLD = 0.5;

// inline std::chrono::high_resolution_clock::time_point
// get_current_time_fenced()
// {
//     std::atomic_thread_fence(std::memory_order_seq_cst);
//     auto res_time = std::chrono::high_resolution_clock::now();
//     std::atomic_thread_fence(std::memory_order_seq_cst);
//     return res_time;
// }


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

#ifdef GPU_IMPLEMENTATION
    size_t NUMBER_OF_THREADS = 16;
    thread_pool pool1(NUMBER_OF_THREADS);
    const std::string kernel_filename = "/home/julfy1/Documents/4th_term/ACS/ACS_Visual_Odometry/kernels/feature_extraction_kernel_functions.bin";
    const auto program = create_platform_from_binary(kernel_filename);

    const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    const auto& device = devices.front(); // TODO Take the best device
    const auto context = program.getInfo<CL_PROGRAM_CONTEXT>();

    const GPU_settings GPU_settings({program, device, context});


#endif


    // std::vector<std::vector<uint8_t>> descs1 = descriptor_for_s_pop(argv[1]);
    // std::vector<std::vector<uint8_t>> descs2 = descriptor_for_s_pop(argv[2]);

    auto start_feature_extraction = get_current_time_fenced();

#ifdef PARALLEL_IMPLEMENTATION
    cv::Mat image1 = cv::imread(argv[1]);
    cv::Mat image2 = cv::imread(argv[2]);

    auto descs1 = feature_extraction_manager_with_points(image1, pool1);
    auto descs2 = feature_extraction_manager_with_points(image2, pool1);



#elif GPU_IMPLEMENTATION
    auto descs1 = feature_extraction_manager_with_points(img1, GPU_settings);
    auto descs2 = feature_extraction_manager_with_points(img2, GPU_settings);


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


// #ifdef VISUALIZATION
//     cv::Mat binaryMatchesImg;
//     cv::drawMatches(img1, std::get<1>(descs1), img2, std::get<1>(descs2), convertToDMatch(customMatches), binaryMatchesImg);
//     cv::imshow("FREAK Matches", binaryMatchesImg);
//     cv::imwrite("FREAK_matches_not_changed.jpeg", binaryMatchesImg);
//     cv::waitKey(0);
// #endif




#ifdef VISUALIZATION

    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : customMatches) {
        points1.push_back(std::get<1>(descs1)[match.first].pt);
        points2.push_back(std::get<1>(descs2)[match.second].pt);
    }


    std::vector<unsigned char> inlierMask;
    cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 1.0, inlierMask);

  
    int rows = std::max(img1.rows, img2.rows);
    int cols = img1.cols + img2.cols;

    cv::Mat inlierCanvas(rows, cols, CV_8UC3);
    cv::Mat outlierCanvas(rows, cols, CV_8UC3);


    if (img1.channels() == 1) cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
    if (img2.channels() == 1) cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);

    img1.copyTo(inlierCanvas(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(inlierCanvas(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    img1.copyTo(outlierCanvas(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(outlierCanvas(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));


    int inlierCount = 0, outlierCount = 0;
    for (size_t i = 0; i < customMatches.size(); ++i) {
        cv::Point2f pt1 = std::get<1>(descs1)[customMatches[i].first].pt;
        cv::Point2f pt2 = std::get<1>(descs2)[customMatches[i].second].pt + cv::Point2f((float)img1.cols, 0);

        if (inlierMask[i]) {
            inlierCount++;

            cv::line(inlierCanvas, pt1, pt2, cv::Scalar(0, 255, 0), 1);
        } else {
            outlierCount++;

            cv::line(outlierCanvas, pt1, pt2, cv::Scalar(0, 0, 255), 1);
        }
    }


    std::cout << "RANSAC inliers: " << inlierCount << ", outliers: " << outlierCount<< std::endl;
    cv::imshow("RANSAC Inliers (Green)", inlierCanvas);
    cv::imshow("RANSAC Outliers (Red)", outlierCanvas);
    cv::imwrite("inliers_green_lines.jpeg", inlierCanvas);
    cv::imwrite("outliers_red_lines.jpeg", outlierCanvas);
    cv::waitKey(0);
#endif


    return 0;
}
