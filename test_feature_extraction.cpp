//
// Created by julfy1 on 5/4/25.
//
#include "feature_extraction_parallel_GPU/feature_extraction_parallel_GPU.h"
#include "feature_extraction_parallel/feature_extraction_parallel.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <CL/cl2.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./matching <image1> <image2>" << std::endl;
        return -1;
    }

    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    auto image_gpu = img1;
    // cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    // GPU

    const std::string kernel_filename = "/home/julfy1/Documents/4th_term/ACS/ACS_Visual_Odometry/kernels/feature_extraction_kernel_functions.bin";
    const auto program = create_platform_from_binary(kernel_filename);

    const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    const auto& device = devices.front();
    const auto context = program.getInfo<CL_PROGRAM_CONTEXT>();

    const GPU_settings GPU_settings({program, device, context});


    cv::Mat my_blurred_gray_gpu;
    cv::GaussianBlur(image_gpu, my_blurred_gray_gpu, cv::Size(7, 7), 0);

    const int n_rows = my_blurred_gray_gpu.rows;
    const int n_cols = my_blurred_gray_gpu.cols;

    std::vector<std::vector<float>> R_score_GPU(n_rows, std::vector<float>(n_cols));

    CornerDetectionParallel_GPU::shitomasi_corner_detection(GPU_settings, my_blurred_gray_gpu, R_score_GPU);


    // Thread pool

    const size_t BLOCK_SIZE = 200;
    size_t NUMBER_OF_THREADS = 16;
    thread_pool pool1(NUMBER_OF_THREADS);
    auto image_tp = img1;

    cv::Mat my_blurred_gray;

    cv::GaussianBlur(image_tp, my_blurred_gray, cv::Size(7, 7), 0);

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
        const interval interval = {{0, n_cols}, {i, std::min(static_cast<int>(i + BLOCK_SIZE + 4), n_rows)}};
        // intervals.emplace_back(interval);
#ifdef VISUALIZATION
        cv::rectangle(image,
                  cv::Point(interval.cols.start, interval.rows.start),
                  cv::Point(interval.cols.end, interval.rows.end),
                  cv::Scalar(0, 255, 0), 1);
#endif
        // print_interval(interval);
        futures_responses.emplace_back(pool1.submit([&my_blurred_gray, &Jx, &Jy, &Jxy, &R_array, interval]() {
            response_worker(my_blurred_gray, interval, Jx, Jy, Jxy, R_array);
        }));
    }

    for (auto &future : futures_responses) {
        future.get();
    }

    std::cout << "HI";

}
