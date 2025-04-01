//
// Created by julfy1 on 3/31/25.
//

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "feature_extraction_parallel/threadpool.h"
#include "feature_extraction_parallel/corner_detection_parallel.h"
#include "feature_extraction_parallel/FREAK_feature_descriptor_parallel.h"
#include "feature_extraction/test_feature_extraction.h"

// #define VISUALIZATION

void response_worker(const cv::Mat& blurred_gray, const interval& interval, cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, std::vector<std::vector<double>>& R_array) {

    CornerDetectionParallel::direction_gradients_worker(blurred_gray, interval, Jx, Jy, Jxy);

    CornerDetectionParallel::shitomasi_corner_detection_worker(Jx, Jy, Jxy, interval, 0.05, R_array);


}




std::vector<std::vector<uint8_t>> feature_extraction_manager(cv::Mat& image, thread_pool& pool) {

    // Preprocess the image and prepare the enviroment

    cv::Mat my_blurred_gray;

    const cv::Mat blurred = CornerDetectionParallel::custom_bgr2gray(image);

    cv::GaussianBlur(blurred, my_blurred_gray, cv::Size(7, 7), 0);

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;
    // rows 400 cols 900
    cv::Mat Jx = cv::Mat::zeros(n_rows, n_cols, CV_64F);
    cv::Mat Jy = cv::Mat::zeros(n_rows, n_cols, CV_64F);
    cv::Mat Jxy = cv::Mat::zeros(n_rows, n_cols, CV_64F);

    std::vector<std::vector<double>> R_array(n_rows, std::vector<double>(n_cols, 0));


    std::vector<std::future<void>> futures_responses;
    std::vector<std::future<void>> futures_descriptor;


    for (int i = 0; i < n_rows; i += BLOCK_SIZE) {
        for (int j = 0; j < n_cols; j += BLOCK_SIZE) {
            const interval interval = {{j, std::min(n_cols, j + BLOCK_SIZE - 1)}, {i, std::min(n_rows, i + BLOCK_SIZE - 1)}};
            // intervals.emplace_back(interval);

            futures_responses.emplace_back(pool.submit([&image, &Jx, &Jy, &Jxy, &R_array, interval]() {
                response_worker(image, interval, Jx, Jy, Jxy, R_array);
            }));
        }
    }

    for (auto &future : futures_responses) {
        future.get();
    }

    const auto local_mins_shitomasi = CornerDetectionParallel::non_maximum_suppression(R_array, n_rows, n_cols, 5, 15000);

    size_t cur_index = 0;
    const size_t num_of_keypoints = local_mins_shitomasi.size();

    std::vector<std::vector<uint8_t>> descriptor(num_of_keypoints, std::vector<uint8_t>(DESCRIPTOR_SIZE));


    while (cur_index + KEY_POINTS_PER_TASK < num_of_keypoints) {

        futures_descriptor.emplace_back(pool.submit([&local_mins_shitomasi, &image, cur_index, &descriptor, &num_of_keypoints]() {
            FREAK_Parallel::FREAK_feature_description_worker(local_mins_shitomasi, image, cur_index, descriptor, num_of_keypoints);
        }));
        cur_index += KEY_POINTS_PER_TASK;
    }

    futures_descriptor.emplace_back(pool.submit([&local_mins_shitomasi, &image, cur_index, &descriptor, &num_of_keypoints]() {
            FREAK_Parallel::FREAK_feature_description_worker(local_mins_shitomasi, image, cur_index, descriptor, num_of_keypoints, num_of_keypoints - cur_index);
        }));

    for (auto &future : futures_descriptor) {
        future.get();
    }

    return descriptor;



}







int main() {
    const std::string filename = "/home/julfy1/Documents/4th_term/ACS/ACS_Visual_Odometry_SOFIA/ACS_Visual_Odometry/images/Zhovkva2.jpg";

    cv::Mat image = cv::imread(filename);


    thread_pool pool(NUMBER_OF_THREADS);

    auto descriptor = feature_extraction_manager(image, pool);

    print_descriptor(descriptor);




    #ifdef VISUALIZATION
    for (auto coords : local_mins_shitomasi) {

            // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

            cv::circle(image1, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
    }

    cv::imshow("BOhdan with corners harris", image1);

    cv::waitKey(0);

    cv::destroyAllWindows();
    #endif

}