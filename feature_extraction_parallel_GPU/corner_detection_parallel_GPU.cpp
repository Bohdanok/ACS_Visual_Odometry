//
// Created by julfy1 on 2/1/25.
//

#include "corner_detection_parallel_GPU.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <cmath>
#include <iostream>
#include <queue>

// #define INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK

cv::Mat CornerDetectionParallel_GPU::custom_bgr2gray(cv::Mat& picture) {
    const int n_rows = picture.rows;
    const int n_cols = picture.cols * 3;
    cv::Mat output_picture = cv::Mat::zeros(n_rows, n_cols / 3, CV_8UC1);

    for (int i = 0; i < n_rows; i++) {
        const auto *ptr_src = picture.ptr<uchar>(i);
        auto *ptr_dst = output_picture.ptr<uchar>(i);
        for (int j = 0; j < n_cols; j += 3) {
            ptr_dst[j / 3] = cv::saturate_cast<uchar>(0.114 * ptr_src[j] + 0.587 * ptr_src[j + 1] + 0.299 * ptr_src[j + 2]);
        }
    }
    return output_picture;

}

void CornerDetectionParallel_GPU::shitomasi_corner_detection(const GPU_settings& gpu_settings, const cv::Mat& my_blurred_gray, std::vector<std::vector<double>>& R_score) {

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;
#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    // long long buffer_write_time = 0;
    // const auto end = get_current_time_fenced();
    // std::cout << "GPU response calculations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //           << " ms" << std::endl;
    const auto start_buffer_write = get_current_time_fenced();
#endif

    const cl::Buffer image_buffer(gpu_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, n_cols * n_rows * sizeof(uchar), my_blurred_gray.data);

    const cl::Buffer Jx_buffer(gpu_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));
    const cl::Buffer Jy_buffer(gpu_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));
    const cl::Buffer Jxy_buffer(gpu_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));
#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_buffer_write = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "Buffer write time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_buffer_write - start_buffer_write).count()
              << " ms" << std::endl;
#endif


    cl::Kernel kernel_gradient(gpu_settings.program, "gradient_convolution");

    kernel_gradient.setArg(0, image_buffer);
    kernel_gradient.setArg(1, Jx_buffer);
    kernel_gradient.setArg(2, Jy_buffer);
    kernel_gradient.setArg(3, Jxy_buffer);
    kernel_gradient.setArg(4, n_cols);


    const cl::CommandQueue command_queue(gpu_settings.context, gpu_settings.device);

    const cl::NDRange global_offset_gradient(1, 1);
    const cl::NDRange global_size_gradient(n_cols - 2, n_rows - 2);

    command_queue.enqueueNDRangeKernel(kernel_gradient, global_offset_gradient, global_size_gradient);

    command_queue.finish();

    // R score

    cl::Kernel kernel_shitomasi_response(gpu_settings.program, "shitomasi_response");

    cl::Buffer R_response_buffer(gpu_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));


    kernel_shitomasi_response.setArg(0, R_response_buffer);
    kernel_shitomasi_response.setArg(1, Jx_buffer);
    kernel_shitomasi_response.setArg(2, Jy_buffer);
    kernel_shitomasi_response.setArg(3, Jxy_buffer);
    kernel_shitomasi_response.setArg(4, n_cols);
    kernel_shitomasi_response.setArg(5, RESPONSE_THRESHOLD);


    const cl::NDRange global_offset_response(2, 2);
    const cl::NDRange global_size_response(n_cols - 4, n_rows - 4);

    command_queue.enqueueNDRangeKernel(kernel_shitomasi_response, global_offset_response, global_size_response);

    command_queue.finish();

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_kernel_functions = get_current_time_fenced();
    std::cout << "GPU kernel function calculations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_kernel_functions - end_buffer_write).count()
              << " ms" << std::endl;
#endif

    // std::vector<std::vector<double>> R_array(n_rows, std::vector<double>(n_cols, 0));
    //
    // command_queue.enqueueReadBuffer(R_response_buffer, CL_TRUE, 0, sizeof(double) * n_rows * n_cols, R_array.data());

    std::vector<double> flat_R_array(n_rows * n_cols);
#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto start_buffer_read = get_current_time_fenced();
#endif

    command_queue.enqueueReadBuffer(R_response_buffer, CL_TRUE, 0, sizeof(double) * flat_R_array.size(), flat_R_array.data());


#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_buffer_read = get_current_time_fenced();
    std::cout << "Buffer read time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_buffer_read - start_buffer_read).count()
              << " ms" << std::endl;
#endif

    // std::vector<std::vector<double>> R_array(n_rows, std::vector<double>(n_cols));
#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto crutch_start = get_current_time_fenced();
#endif
    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_cols; ++j)
            R_score[i][j] = flat_R_array[i * n_cols + j];

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto crutch_end = get_current_time_fenced();
    std::cout << "Crutch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(crutch_end - crutch_start).count()
              << " ms" << std::endl;
#endif
    // auto end = get_current_time_fenced();
    // std::cout << "GPU gradient calculations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //           << " ms" << std::endl;


}



std::vector<cv::KeyPoint> CornerDetectionParallel_GPU::non_maximum_suppression(const std::vector<std::vector<double>> &R_values, const int& n_rows, const int& n_cols, const int& k, const int& N) {
    std::priority_queue<std::tuple<double, int, int>> max_heap; // Store (R_value, i, j)
    std::vector<cv::KeyPoint> output_corners;
    output_corners.reserve(N);
    int count = 0;

    for (int i = k / 2; i < n_rows - k / 2; i++) {
        for (int j = k / 2; j < n_cols - k / 2; j++) {

            // not to include out of bounce for retinal sampling

            if (!((j >= 37) && (j <= n_cols - 37) && (i >= 35) && (i <= n_rows - 35))) {
                continue;
            }

            double center_val = R_values[i][j];
            bool is_local_max = true;

            for (int n = i - k / 2; n <= i + k / 2; n++) {
                for (int m = j - k / 2; m <= j + k / 2; m++) {
                    if (!(n == i && m == j)) {
                        if (R_values[n][m] >= center_val) {
                            is_local_max = false;
                            break;
                        }
                    }
                }
                if (!is_local_max) break;
            }

            if (is_local_max) {
                max_heap.push({center_val, i, j});
            }
        }
    }

    for (int i = 0; i < N && !max_heap.empty(); i++) {
        output_corners.push_back({cv::Point2f(static_cast<float>(std::get<2>(max_heap.top())), static_cast<float>(std::get<1>(max_heap.top()))), 1.0f});
        max_heap.pop();
        count++;
    }
    return output_corners;
}


