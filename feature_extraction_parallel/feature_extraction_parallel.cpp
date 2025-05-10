//
// Created by julfy1 on 4/1/25.
//

#include "feature_extraction_parallel.h"

#include <iostream>
#include <opencv2/highgui.hpp>

// #define VISUALIZATION
// #define INTERMEDIATE_TIME_MEASUREMENT

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

void print_interval(const interval& interval) {
    std::cout << "Start: (" << interval.cols.start << ", " << interval.rows.start << ")\tEnd: (" << interval.cols.end << ", " << interval.rows.end << ")" << std::endl;
}


void draw_score_distribution(const std::vector<std::vector<float>>& R_values, const std::string& win_name) {

    int rows = R_values.size();
    int cols = R_values[0].size();

    cv::Mat mat(rows, cols, CV_32F); // Create matrix to store values

    // Copy values
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat.at<float>(i, j) = R_values[i][j];
        }
    }

    // Normalize values to range [0, 1]
    float minVal, maxVal;
    cv::minMaxLoc(mat, reinterpret_cast<double*>(&minVal), reinterpret_cast<double*>(&maxVal));
    std::cout << "Min R: " << minVal << ", Max R: " << maxVal << std::endl;

    // cv::Mat normMat = (mat - minVal) / (maxVal - minVal); // Normalize between 0-1

    float visMax = 300000; // Experiment with this
    cv::Mat normMat = mat / visMax;
    cv::threshold(normMat, normMat, 1.0, 1.0, cv::THRESH_TRUNC);

    // Create color image
    cv::Mat colorImage(rows, cols, CV_8UC3);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = normMat.at<float>(i, j); // Normalized value [0, 1]

            // Map to RGB colors (Green → Blue → Red)
            uchar red   = static_cast<uchar>(255 * std::max(0.f, (val - 0.5f) * 2));
            uchar blue  = static_cast<uchar>(255 * std::max(0.f, (0.5f - std::abs(val - 0.5f)) * 2));
            uchar green = static_cast<uchar>(255 * std::max(0.f, (0.5f - val) * 2));

            colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        }
    }

    cv::imshow(win_name, colorImage);
    // cv::imwrite("../test_images/output_images/" + win_name + ".png", colorImage);
}

void response_worker(const cv::Mat& blurred_gray, const interval& interval, cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, std::vector<std::vector<float>>& R_array) {

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
    cv::Mat Jx = cv::Mat::zeros(n_rows, n_cols, CV_32F);
    cv::Mat Jy = cv::Mat::zeros(n_rows, n_cols, CV_32F);
    cv::Mat Jxy = cv::Mat::zeros(n_rows, n_cols, CV_32F);

    std::vector<std::vector<float>> R_array(n_rows, std::vector<float>(n_cols, 0));


    std::vector<std::future<void>> futures_responses;
    std::vector<std::future<void>> futures_descriptor;


    for (int i = 0; i < n_rows; i += BLOCK_SIZE) {
        // const interval interval = {{j, std::min(n_cols, j + BLOCK_SIZE + 2)}, {i, std::min(n_rows, i + BLOCK_SIZE + 2)}};
        // const interval interval = {{0, std::min(i + BLOCK_SIZE, n_rows)}, {i, std::min(n_rows, i + BLOCK_SIZE + 2)}};
        const interval interval = {{0, n_cols}, {i, std::min(i + BLOCK_SIZE + 4, n_rows)}};
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

//
//     for (int i = 0; i < n_rows; i += BLOCK_SIZE) {
//         for (int j = 0; j < n_cols; j += BLOCK_SIZE) {
//             // const interval interval = {{j, std::min(n_cols, j + BLOCK_SIZE + 2)}, {i, std::min(n_rows, i + BLOCK_SIZE + 2)}};
//             const interval interval = {{j, std::min(n_cols, j + BLOCK_SIZE + 2)}, {i, std::min(n_rows, i + BLOCK_SIZE + 2)}};
//
//             // intervals.emplace_back(interval);
// #ifdef VISUALIZATION
//             cv::rectangle(image,
//                       cv::Point(interval.cols.start, interval.rows.start),
//                       cv::Point(interval.cols.end, interval.rows.end),
//                       cv::Scalar(0, 255, 0), 1);
// #endif
//             // print_interval(interval);
//             futures_responses.emplace_back(pool.submit([&my_blurred_gray, &Jx, &Jy, &Jxy, &R_array, interval]() {
//                 response_worker(my_blurred_gray, interval, Jx, Jy, Jxy, R_array);
//             }));
//         }
//     }

    for (auto &future : futures_responses) {
        future.get();
    }
    // cv::imshow("Jx", Jx);
    // cv::imshow("Jy", Jy);
    // cv::imshow("Jxy", Jxy);
    //
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    auto local_mins_shitomasi = CornerDetectionParallel::non_maximum_suppression(R_array, n_rows, n_cols, 5, 1500);

    std::sort(local_mins_shitomasi.begin(), local_mins_shitomasi.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        const int ay = static_cast<int>(a.pt.y);
        const int by = static_cast<int>(b.pt.y);
        if (ay == by)
            return static_cast<int>(a.pt.x) < static_cast<int>(b.pt.x);
        return ay < by;
    });


    #ifdef VISUALIZATION
        for (auto coords : local_mins_shitomasi) {

            // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

            cv::circle(image, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
        }

        cv::imshow("BOhdan with corners harris", image);

        draw_score_distribution(R_array, "Response");
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
#ifdef INTERMEDIATE_TIME_MEASUREMENT
    auto start = get_current_time_fenced();
#endif

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;
    // rows 400 cols 900
    cv::Mat Jx = cv::Mat::zeros(n_rows, n_cols, CV_32F);
    cv::Mat Jy = cv::Mat::zeros(n_rows, n_cols, CV_32F);
    cv::Mat Jxy = cv::Mat::zeros(n_rows, n_cols, CV_32F);

    std::vector<std::vector<float>> R_array(n_rows, std::vector<float>(n_cols, 0));


    std::vector<std::future<void>> futures_responses;
    std::vector<std::future<void>> futures_descriptor;


    for (int i = 0; i < n_rows; i += BLOCK_SIZE) {
        // const interval interval = {{j, std::min(n_cols, j + BLOCK_SIZE + 2)}, {i, std::min(n_rows, i + BLOCK_SIZE + 2)}};
        // const interval interval = {{0, std::min(i + BLOCK_SIZE, n_rows)}, {i, std::min(n_rows, i + BLOCK_SIZE + 2)}};
        const interval interval = {{0, n_cols}, {i, std::min(i + BLOCK_SIZE + 4, n_rows)}};
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

    for (auto &future : futures_responses) {
        future.get();
    }
    // cv::imshow("Jx", Jx);
    // cv::imshow("Jy", Jy);
    // cv::imshow("Jxy", Jxy);
#ifdef INTERMEDIATE_TIME_MEASUREMENT
    auto end = get_current_time_fenced();
    std::cout << "Threadpool gradient calculations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
#endif
    // draw_score_distribution(R_array, "Parallel");
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    auto local_mins_shitomasi = CornerDetectionParallel::non_maximum_suppression(R_array, n_rows, n_cols, 5, 1500);

    std::sort(local_mins_shitomasi.begin(), local_mins_shitomasi.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        const int ay = static_cast<int>(a.pt.y);
        const int by = static_cast<int>(b.pt.y);
        if (ay == by)
            return static_cast<int>(a.pt.x) < static_cast<int>(b.pt.x);
        return ay < by;
    });

    // cv::Mat new_image;
    // if (blurred.channels() == 1) cv::cvtColor(blurred, new_image, cv::COLOR_GRAY2BGR);
    //
    // for (auto coords : local_mins_shitomasi) {
    //
    //     // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;
    //
    //     cv::circle(new_image, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
    // }
    // cv::imshow("GPU CORNERS", new_image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

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
#ifdef INTERMEDIATE_TIME_MEASUREMENT
    auto start_descriptor = get_current_time_fenced();
#endif

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
#ifdef INTERMEDIATE_TIME_MEASUREMENT
    auto end_descriptor = get_current_time_fenced();
    std::cout << "Threadpool descriptor calculations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_descriptor - start_descriptor).count()
              << " ms" << std::endl;
#endif
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
