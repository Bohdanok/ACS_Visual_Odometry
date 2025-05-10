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
