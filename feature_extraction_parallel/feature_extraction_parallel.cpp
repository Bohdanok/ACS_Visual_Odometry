//
// Created by julfy1 on 4/1/25.
//

#include "feature_extraction_parallel.h"

#include <iostream>
#include <opencv2/highgui.hpp>

// #define VISUALIZATION

void print_interval(const interval& interval) {
    std::cout << "Start: (" << interval.cols.start << ", " << interval.rows.start << ")\tEnd: (" << interval.cols.end << ", " << interval.rows.end << ")" << std::endl;
}

void draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name) {

    int rows = R_values.size();
    int cols = R_values[0].size();

    cv::Mat mat(rows, cols, CV_64F); // Create matrix to store values

    // Copy values
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat.at<double>(i, j) = R_values[i][j];
        }
    }

    // Normalize values to range [0, 1]
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Mat normMat = (mat - minVal) / (maxVal - minVal); // Normalize between 0-1

    // Create color image
    cv::Mat colorImage(rows, cols, CV_8UC3);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = normMat.at<double>(i, j); // Normalized value [0, 1]

            // Map to RGB colors (Green → Blue → Red)
            uchar red   = static_cast<uchar>(255 * std::max(0.0, (val - 0.5) * 2));  // Increase red for higher values
            uchar blue  = static_cast<uchar>(255 * std::max(0.0, (0.5 - std::abs(val - 0.5)) * 2));  // Max in the middle
            uchar green = static_cast<uchar>(255 * std::max(0.0, (0.5 - val) * 2));  // Decrease green as value increases

            colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        }
    }
    cv::imshow(win_name, colorImage);
    // cv::imwrite("../test_images/output_images/" + win_name + ".png", colorImage);

}

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
            const interval interval = {{j, std::min(n_cols, j + BLOCK_SIZE + 2)}, {i, std::min(n_rows, i + BLOCK_SIZE + 2)}};
            // intervals.emplace_back(interval);
#ifdef VISUALIZATION
            cv::rectangle(image,
                      cv::Point(interval.cols.start, interval.rows.start),
                      cv::Point(interval.cols.end, interval.rows.end),
                      cv::Scalar(0, 255, 0), 1);
#endif
            // print_interval(interval);
            futures_responses.emplace_back(pool.submit([&my_blurred_gray, &Jx, &Jy, &Jxy, &R_array, interval]() {
                response_worker(my_blurred_gray, interval, Jx, Jy, Jxy, R_array);
            }));
        }
    }

    for (auto &future : futures_responses) {
        future.get();
    }

    const auto local_mins_shitomasi = CornerDetectionParallel::non_maximum_suppression(R_array, n_rows, n_cols, 5, 1500);

    #ifdef VISUALIZATION
        for (auto coords : local_mins_shitomasi) {

            // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

            cv::circle(image, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
        }

        cv::imshow("BOhdan with corners harris", image);

        // draw_score_distribution(R_array, "Response");
        //
        // cv::imshow("Jx", Jx);
        // cv::imshow("Jy", Jy);
        // cv::imshow("Jxy", Jxy);

        cv::waitKey(0);

        cv::destroyAllWindows();
    #endif

    size_t cur_index = 0;
    const size_t num_of_keypoints = local_mins_shitomasi.size();

    std::vector<std::vector<uint8_t>> descriptor(num_of_keypoints, std::vector<uint8_t>(DESCRIPTOR_SIZE));


    while (cur_index + KEY_POINTS_PER_TASK < num_of_keypoints) {

        futures_descriptor.emplace_back(pool.submit([&local_mins_shitomasi, &my_blurred_gray, cur_index, &descriptor, &num_of_keypoints]() {
            FREAK_Parallel::FREAK_feature_description_worker(local_mins_shitomasi, my_blurred_gray, cur_index, descriptor, num_of_keypoints);
        }));
        cur_index += KEY_POINTS_PER_TASK;
    }

    futures_descriptor.emplace_back(pool.submit([&local_mins_shitomasi, &my_blurred_gray, cur_index, &descriptor, &num_of_keypoints]() {
            FREAK_Parallel::FREAK_feature_description_worker(local_mins_shitomasi, my_blurred_gray, cur_index, descriptor, num_of_keypoints, num_of_keypoints - cur_index);
        }));

    for (auto &future : futures_descriptor) {
        future.get();
    }

    // cv::imshow("Intbervals", image);
    //
    // cv::waitKey(0);
    //
    // cv::destroyAllWindows();

    return descriptor;

}

std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> feature_extraction_manager_with_points(cv::Mat& image, thread_pool& pool) {

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
            const interval interval = {{j, std::min(n_cols, j + BLOCK_SIZE + 2)}, {i, std::min(n_rows, i + BLOCK_SIZE + 2)}};
            // intervals.emplace_back(interval);
#ifdef VISUALIZATION
            cv::rectangle(image,
                      cv::Point(interval.cols.start, interval.rows.start),
                      cv::Point(interval.cols.end, interval.rows.end),
                      cv::Scalar(0, 255, 0), 1);
            print_interval(interval);

#endif
            futures_responses.emplace_back(pool.submit([&my_blurred_gray, &Jx, &Jy, &Jxy, &R_array, interval]() {
                response_worker(my_blurred_gray, interval, Jx, Jy, Jxy, R_array);
            }));
        }
    }

    for (auto &future : futures_responses) {
        future.get();
    }

    const auto local_mins_shitomasi = CornerDetectionParallel::non_maximum_suppression(R_array, n_rows, n_cols, 5, 1500);

    #ifdef VISUALIZATION
        for (auto coords : local_mins_shitomasi) {

            // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

            cv::circle(image, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
        }

        cv::imshow("BOhdan with corners harris", image);

        // draw_score_distribution(R_array, "Response");
        //
        // cv::imshow("Jx", Jx);
        // cv::imshow("Jy", Jy);
        // cv::imshow("Jxy", Jxy);

        cv::waitKey(0);

        cv::destroyAllWindows();
    #endif

    size_t cur_index = 0;
    const size_t num_of_keypoints = local_mins_shitomasi.size();

    std::vector<std::vector<uint8_t>> descriptor(num_of_keypoints, std::vector<uint8_t>(DESCRIPTOR_SIZE));


    while (cur_index + KEY_POINTS_PER_TASK < num_of_keypoints) {

        futures_descriptor.emplace_back(pool.submit([&local_mins_shitomasi, &my_blurred_gray, cur_index, &descriptor, &num_of_keypoints]() {
            FREAK_Parallel::FREAK_feature_description_worker(local_mins_shitomasi, my_blurred_gray, cur_index, descriptor, num_of_keypoints);
        }));
        cur_index += KEY_POINTS_PER_TASK;
    }

    futures_descriptor.emplace_back(pool.submit([&local_mins_shitomasi, &my_blurred_gray, cur_index, &descriptor, &num_of_keypoints]() {
            FREAK_Parallel::FREAK_feature_description_worker(local_mins_shitomasi, my_blurred_gray, cur_index, descriptor, num_of_keypoints, num_of_keypoints - cur_index);
        }));

    for (auto &future : futures_descriptor) {
        future.get();
    }

    return {descriptor, local_mins_shitomasi};

}




void print_descriptor(const std::vector<std::vector<uint8_t>>& descriptor){
    std::cout << "/////////////////////////////////////////////////////////////" << std::endl;
    for (size_t i = 0; i < descriptor.size(); i++) {
        std::cout << "<";
        for (size_t j = 0; j < descriptor[0].size(); j++) {
            std::cout << static_cast<int>(descriptor[i][j]) << ", ";
        }
        std::cout << ">" << std::endl;
    }
    std::cout << "/////////////////////////////////////////////////////////////" << std::endl;

}

bool is_number(const std::string& s) {
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}


//
// int main() {
//     const std::string filename = "/home/julfy1/Documents/4th_term/ACS/ACS_Visual_Odometry_SOFIA/ACS_Visual_Odometry/images/Zhovkva2.jpg";
//
//     cv::Mat image = cv::imread(filename);
//
//
//     thread_pool pool(NUMBER_OF_THREADS);
//
//     const auto descriptor = feature_extraction_manager(image, pool);
//
//     print_descriptor(descriptor);
//
// #ifdef VISUALIZATION
//     for (auto coords : local_mins_shitomasi) {
//
//         // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;
//
//         cv::circle(image1, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
//     }
//
//     cv::imshow("BOhdan with corners harris", image1);
//
//     cv::waitKey(0);
//
//     cv::destroyAllWindows();
// #endif
//
//
// }
