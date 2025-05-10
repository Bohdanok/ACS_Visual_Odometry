#include "feature_extraction_parallel/threadpool.h"
#include "feature_matching_parallel/feature_matching_parallel.h"
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
#elif GPU_IMPLEMENTATION
#include "feature_extraction_parallel_GPU/feature_extraction_parallel_GPU.h"
#else

    #include "feature_extraction/test_feature_extraction.h"
#endif


#define VISUALIZATION
#ifndef GPU_IMPLEMENTATION

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}
#endif

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

    NUMBER_OF_THREADS = std::stoi(argv[3]);
    BLOCK_SIZE = std::stoi(argv[4]);

    // std::cout << "Number of threads: " << NUMBER_OF_THREADS << std::endl;
    // std::cout << "Block size: " << BLOCK_SIZE << std::endl;

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

    thread_pool pool1(NUMBER_OF_THREADS);
    // thread_pool pool2(NUMBER_OF_THREADS);

    auto descs1 = feature_extraction_manager_with_points(image1, pool1);
    auto descs2 = feature_extraction_manager_with_points(image2, pool1);



#elif GPU_IMPLEMENTATION
    auto descs1 = feature_extraction_manager_with_points(img1, GPU_settings);
    auto descs2 = feature_extraction_manager_with_points(img2, GPU_settings);

    // print_descriptor(descs1.first);

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
    customMatches = matchCustomBinaryDescriptorsThreadPool(descs1.first, descs2.first, pool1, NUMBER_OF_THREADS, 0.75f);
 //    customMatches = matchCustomBinaryDescriptorsThreadPool(
 //    	std::get<0>(descs1),
 //    	std::get<0>(descs2),
 //    	pool1,
 //    	NUMBER_OF_THREADS
	// );

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