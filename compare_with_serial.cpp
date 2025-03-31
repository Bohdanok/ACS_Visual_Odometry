//
// Created by julfy1 on 3/31/25.
//

#include "feature_extraction_parallel/threadpool.h"
#include "feature_extraction_parallel/corner_detection_parallel.h"
#include "feature_extraction/test_feature_extraction.h"


std::vector<cv::KeyPoint> compute_key_points_in_block(const cv::Mat& blurred_gray, const interval& interval) {
    std::vector<cv::KeyPoint> key_points;

    const std::vector<cv::Mat> gradients = CornerDetection::direction_gradients_worker(blurred_gray, interval);

    const std::vector<std::vector<double>> shitomasi_response = CornerDetection::shitomasi_corner_detection_worker(gradients[0], gradients[1], gradients[2], interval, 0.05);

    // const

    return key_points;

}

auto test_response(const cv::Mat& blurred_gray, const interval& interval) {

    const std::vector<cv::Mat> gradients = CornerDetection::direction_gradients_worker(blurred_gray, interval);

    const std::vector<std::vector<double>> shitomasi_response = CornerDetection::shitomasi_corner_detection_worker(gradients[0], gradients[1], gradients[2], interval, 0.05);

    return shitomasi_response;

}



auto compute_keypoints_parallel(const cv::Mat& image, const size_t& n_cols, const size_t& n_rows, thread_pool& pool) {
    std::vector<std::future<std::vector<cv::KeyPoint>>> key_points;

    for (size_t i = 0; i < n_rows; i += BLOCK_SIZE) {
        for (size_t j = 0; j < n_cols; j += BLOCK_SIZE) {

            const auto interval = {{i, std::min(n_rows, i + BLOCK_SIZE - 1)}, {j, std::min(n_cols, j + BLOCK_SIZE - 1)}};

            key_points.push_back(pool.submit([&image, interval]() {
                return compute_key_points_in_block(image, interval);
            }));

        }
    }

    return key_points;
}







int main() {
    const std::string filename = "/home/julfy1/Documents/4th_term/ACS/ACS_Visual_Odometry_SOFIA/ACS_Visual_Odometry/images/Zhovkva2.jpg";

    cv::Mat image = cv::imread(filename);

    cv::Mat my_blurred_gray;

    const cv::Mat blurred = CornerDetection::custom_bgr2gray(image);

    cv::GaussianBlur(blurred, my_blurred_gray, cv::Size(7, 7), 0);


    const size_t n_rows = my_blurred_gray.rows;
    const size_t n_cols = my_blurred_gray.cols;

    thread_pool pool(NUMBER_OF_THREADS);

    auto key_points_futures = compute_keypoints_parallel(my_blurred_gray, n_cols, n_rows, pool);

    std::vector<cv::KeyPoint> key_points;
    for (auto &future : key_points_futures) {
        std::vector<cv::KeyPoint> block_corners = future.get();
        key_points.insert(key_points.end(), block_corners.begin(), block_corners.end());
    }

}