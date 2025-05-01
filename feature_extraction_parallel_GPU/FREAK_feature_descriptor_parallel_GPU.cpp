//
// Created by julfy1 on 3/24/25.
//

#include "FREAK_feature_descriptor_parallel_GPU.h"
#include <iostream>

#define INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK

std::vector<std::vector<uint8_t>> FREAK_Parallel_GPU::FREAK_feature_description(const std::vector<cv::KeyPoint>& key_points, const cv::Mat& blurred_gray_picture, const GPU_settings& GPU_settings) {
    const size_t num_of_keypoints = key_points.size();
    std::vector<point> key_points_to_read(num_of_keypoints);
    std::vector<std::vector<uint8_t>> descriptor(num_of_keypoints, std::vector<uint8_t>(DESCRIPTOR_SIZE));
    std::vector<uint8_t> flat_descriptor(num_of_keypoints * DESCRIPTOR_SIZE, 5);

    // uint8_t** descriptor = new uint8_t*[num_of_keypoints];
    const int n_rows = blurred_gray_picture.rows;
    const int n_cols = blurred_gray_picture.cols;
#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto compatibility_start = get_current_time_fenced();
#endif

    for (size_t i = 0; i < num_of_keypoints; i++) {

        key_points_to_read[i] = point{static_cast<int>(key_points[i].pt.x), static_cast<int>(key_points[i].pt.y)};
        // key_points_to_read[i][1] = static_cast<int>(key_points[i].pt.y);

    }
#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_compatibility = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "cv::KeyPoint compatibility crutch: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_compatibility - compatibility_start).count()
              << " ms" << std::endl;
#endif

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto buffer_write_start = get_current_time_fenced();
#endif

    const cl::Buffer descriptor_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(uint8_t) * num_of_keypoints * DESCRIPTOR_SIZE);
    const cl::Buffer image_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, n_cols * n_rows * sizeof(uchar), blurred_gray_picture.data);

    const cl::Buffer test_cases_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(test) * NUM_PAIRS, test_cases.data());
    // const cl::Buffer patch_description_points_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(size_t) * DESCRIPTOR_SIZE, PATCH_DESCRIPTION_POINTS.data());
    const cl::Buffer patch_description_points_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(size_t) * DESCRIPTOR_SIZE, static_cast<void*>(const_cast<size_t*>(PATCH_DESCRIPTION_POINTS.data())));

    const cl::Buffer key_points_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(point) * num_of_keypoints, key_points_to_read.data());
    // orientation
    // const cl::Buffer angle_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float));
    const cl::Buffer rotation_matrix_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float) * 4);

    const cl::Buffer O_x_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float));
    const cl::Buffer O_y_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float));

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_buffer_write = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "FREAK buffer write time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_buffer_write - buffer_write_start).count()
              << " ms" << std::endl;
#endif

    const cl::CommandQueue command_queue(GPU_settings.context, GPU_settings.device);


    cl::Kernel orientation_kernel(GPU_settings.program, "compute_orientation");
    cl::Kernel merge_orientation_kernel(GPU_settings.program, "merge_orientation_tasks");
    cl::Kernel compute_descriptor_kernel(GPU_settings.program, "compute_descriptor");

    for (int i = 0; i < num_of_keypoints; i++) {
        // const float O_x = 0.0f;
        // const float O_y = 0.0f;


        orientation_kernel.setArg(0, image_buffer);
        orientation_kernel.setArg(1, test_cases_buffer);

        orientation_kernel.setArg(2, O_x_buffer);
        orientation_kernel.setArg(3, O_y_buffer);


        orientation_kernel.setArg(4, key_points_buffer);
        orientation_kernel.setArg(5, i); // communication hz
        orientation_kernel.setArg(6, n_cols);

        command_queue.enqueueNDRangeKernel(orientation_kernel, cl::NullRange, cl::NDRange(test_cases.size()));

        command_queue.finish();

        merge_orientation_kernel.setArg(0, O_x_buffer);
        merge_orientation_kernel.setArg(1, O_y_buffer);
        merge_orientation_kernel.setArg(2, rotation_matrix_buffer);


        command_queue.enqueueNDRangeKernel(merge_orientation_kernel, cl::NullRange, cl::NDRange(1));

        command_queue.finish();


        compute_descriptor_kernel.setArg(0, descriptor_buffer);
        compute_descriptor_kernel.setArg(1, image_buffer);
        compute_descriptor_kernel.setArg(2, patch_description_points_buffer);
        compute_descriptor_kernel.setArg(3, key_points_buffer);
        compute_descriptor_kernel.setArg(4, i); // communication hz
        compute_descriptor_kernel.setArg(5, rotation_matrix_buffer);
        compute_descriptor_kernel.setArg(6, test_cases_buffer);
        compute_descriptor_kernel.setArg(7, n_cols);

        command_queue.enqueueNDRangeKernel(compute_descriptor_kernel, cl::NullRange, cl::NDRange(DESCRIPTOR_SIZE));

        command_queue.finish();
        // break;
        // assert(err1 == 0 && err2 == 0 && err3 == 0);
        // if (err1 != CL_SUCCESS && err2 != CL_SUCCESS && err3 != CL_SUCCESS) {
        //     std::cerr << "Error" << std::endl;
        // }
        // command_queue.enqueueReadBuffer(descriptor_buffer, CL_TRUE, 0, sizeof(uint8_t) * flat_descriptor.size(), flat_descriptor.data());
        //
        // if (err4 != CL_SUCCESS) {
        //     std::cerr << "Error reading descriptor buffer! Error: " << err4 << "!" << std::endl;
        //     std::cerr << "Errored index: " << i << std::endl;
        // }
    }

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto buffer_read_start = get_current_time_fenced();
#endif
    const auto err4 = command_queue.enqueueReadBuffer(descriptor_buffer, CL_TRUE, 0, sizeof(uint8_t) * flat_descriptor.size(), flat_descriptor.data());

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_buffer_read = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "FREAK buffer read time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_buffer_read - buffer_read_start).count()
              << " ms" << std::endl;
#endif

    if (err4 != CL_SUCCESS) {
        std::cerr << "Error reading descriptor buffer! Error: " << err4 << "!" << std::endl;
    }

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto crutch_start = get_current_time_fenced();
#endif

    for (int i = 0; i < num_of_keypoints; ++i)
        for (int j = 0; j < DESCRIPTOR_SIZE; ++j)
            descriptor[i][j] = flat_descriptor[i * DESCRIPTOR_SIZE + j];

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto crutch_end = get_current_time_fenced();
    std::cout << "Crutch time: " << std::chrono::duration_cast<std::chrono::milliseconds>(crutch_end - crutch_start).count()
              << " ms" << std::endl;
#endif

    return descriptor;

}
