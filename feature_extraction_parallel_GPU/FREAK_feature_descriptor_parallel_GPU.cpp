//
// Created by julfy1 on 3/24/25.
//

#include "FREAK_feature_descriptor_parallel_GPU.h"
#include <iostream>

// #define INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK

std::vector<std::vector<uint8_t>> FREAK_Parallel_GPU::FREAK_feature_description(const std::vector<cv::KeyPoint>& key_points, const cv::Mat& blurred_gray_picture, const GPU_settings& GPU_settings) {
    const size_t num_of_keypoints = key_points.size();
    std::vector<std::array<int, 2>> key_points_to_read(num_of_keypoints);
    std::vector<std::vector<uint8_t>> descriptor(num_of_keypoints, std::vector<uint8_t>(DESCRIPTOR_SIZE));
    std::vector<uint8_t> flat_descriptor(num_of_keypoints * DESCRIPTOR_SIZE, 5);

    // uint8_t** descriptor = new uint8_t*[num_of_keypoints];
    const int n_rows = blurred_gray_picture.rows;
    const int n_cols = blurred_gray_picture.cols;
#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    std::cout << "\n===============================" << std::endl;
    const auto compatibility_start = get_current_time_fenced();
#endif

    for (size_t i = 0; i < num_of_keypoints; i++) {

        key_points_to_read[i] = {{static_cast<int>(key_points[i].pt.x), static_cast<int>(key_points[i].pt.y)}};
        // std::cout << "Key Point: (" << key_points_to_read[i][0] << ", " << key_points_to_read[i][1] << ")" << std::endl;
        // key_points_to_read[i][1] = static_cast<int>(key_points[i].pt.y);

    }
    // std::cout << "CPU (349, 37) = " << (int)blurred_gray_picture.at<uchar>(349, 37) << std::endl;

    // right after static test_cases is initialized:
    // for (size_t k = 0; k < 10; ++k) {
    //     auto &t = test_cases[k];
    //     std::cout << "Host Test num: " << k
    //               << "   point1 = (" << t.point1.x << "," << t.point1.y << ")"
    //               << "   point2 = (" << t.point2.x << "," << t.point2.y << ")\n";
    // }


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

    const cl::Buffer test_cases_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * 4 * test_cases.size(), test_cases.data());

    // // CORRECT: copy the host array into GPU memory immediately
    // cl::Buffer test_cases_buffer(
    //     GPU_settings.context,
    //     CL_MEM_READ_ONLY
    //   | CL_MEM_COPY_HOST_PTR,
    //     sizeof(test) * test_cases.size(),
    //     test_cases.data()
    // );

    // const cl::Buffer patch_description_points_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(size_t) * DESCRIPTOR_SIZE, PATCH_DESCRIPTION_POINTS.data());
    const cl::Buffer patch_description_points_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(size_t) * DESCRIPTOR_SIZE, static_cast<void*>(const_cast<size_t*>(PATCH_DESCRIPTION_POINTS.data())));

    const cl::Buffer key_points_buffer(GPU_settings.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * 2 * num_of_keypoints, key_points_to_read.data());
    // orientation
    // const cl::Buffer angle_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float));
    // const cl::Buffer rotation_matrix_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float) * 4 * num_of_keypoints);
    const cl::Buffer rotation_matrix_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float) * 4 * num_of_keypoints);


    const cl::Buffer O_x_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float) * num_of_keypoints);
    const cl::Buffer O_y_buffer(GPU_settings.context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(float) * num_of_keypoints);



#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_buffer_write = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "FREAK buffer write time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_buffer_write - buffer_write_start).count()
              << " ms" << std::endl;
#endif

    const cl::CommandQueue command_queue(GPU_settings.context, GPU_settings.device);
    //
    // command_queue.enqueueFillBuffer(O_x_buffer, 0.0f,0, sizeof(float) * num_of_keypoints);
    // command_queue.enqueueFillBuffer(O_y_buffer, 0.0f,0, sizeof(float) * num_of_keypoints);
    //
    // command_queue.finish();


    // cl::Kernel orientation_kernel(GPU_settings.program, "compute_orientation");
    // cl::Kernel merge_orientation_kernel(GPU_settings.program, "merge_orientation_tasks");
    // cl::Kernel compute_descriptor_kernel(GPU_settings.program, "compute_descriptor");



    cl::Kernel all_orientation_kernel(GPU_settings.program, "compute_all_orientations");
    cl::Kernel all_orientation_merge_kernel(GPU_settings.program, "merge_all_orientations");
    cl::Kernel all_descriptors_kernel(GPU_settings.program, "compute_all_descriptors");

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto all_orientation_start = get_current_time_fenced();
#endif
    all_orientation_kernel.setArg(0, image_buffer);
    all_orientation_kernel.setArg(1, test_cases_buffer);
    all_orientation_kernel.setArg(2, O_x_buffer);
    all_orientation_kernel.setArg(3, O_y_buffer);
    all_orientation_kernel.setArg(4, key_points_buffer);
    all_orientation_kernel.setArg(5, n_cols);

    command_queue.enqueueNDRangeKernel(all_orientation_kernel, cl::NullRange, cl::NDRange(num_of_keypoints, NUM_PAIRS));

    auto err_orientations = command_queue.finish();

    // std::cout << "Error orientation: " << err_orientations << std::endl;

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_all_orientation = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "All orientation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_all_orientation - all_orientation_start).count()
              << " ms" << std::endl;
#endif

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto all_orientation_merge_start = get_current_time_fenced();
#endif

    all_orientation_merge_kernel.setArg(0, O_x_buffer);
    all_orientation_merge_kernel.setArg(1, O_y_buffer);
    all_orientation_merge_kernel.setArg(2, rotation_matrix_buffer);


    command_queue.enqueueNDRangeKernel(all_orientation_merge_kernel, cl::NullRange, cl::NDRange(num_of_keypoints));

    auto err_merge_orientation = command_queue.finish();




#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_all_orientation_merge = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "All orientation merge time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_all_orientation_merge - all_orientation_merge_start).count()
              << " ms" << std::endl;
#endif


#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto all_descriptors_start = get_current_time_fenced();
#endif
    all_descriptors_kernel.setArg(0, descriptor_buffer);
    all_descriptors_kernel.setArg(1, image_buffer);
    all_descriptors_kernel.setArg(2, patch_description_points_buffer);
    all_descriptors_kernel.setArg(3, key_points_buffer);
    all_descriptors_kernel.setArg(4, rotation_matrix_buffer);
    all_descriptors_kernel.setArg(5, test_cases_buffer);
    all_descriptors_kernel.setArg(6, n_cols);

    command_queue.enqueueNDRangeKernel(all_descriptors_kernel, cl::NullRange, cl::NDRange(num_of_keypoints, DESCRIPTOR_SIZE));

    auto err_descriptor = command_queue.finish();

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_all_descriptors = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "All descriptors time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_all_descriptors - all_descriptors_start).count()
              << " ms" << std::endl;
#endif

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
    std::cout << "\n===============================" << std::endl;

#endif

    return descriptor;

}


    // for (int i = 0; i < num_of_keypoints; i++) {
    //     // const float O_x = 0.0f;
    //     // const float O_y = 0.0f;
    //
    //
    //     orientation_kernel.setArg(0, image_buffer);
    //     orientation_kernel.setArg(1, test_cases_buffer);
    //
    //     orientation_kernel.setArg(2, O_x_buffer);
    //     orientation_kernel.setArg(3, O_y_buffer);
    //
    //
    //     orientation_kernel.setArg(4, key_points_buffer);
    //     orientation_kernel.setArg(5, i); // communication hz
    //     orientation_kernel.setArg(6, n_cols);
    //
    //     command_queue.enqueueNDRangeKernel(orientation_kernel, cl::NullRange, cl::NDRange(test_cases.size()));
    //
    //     command_queue.finish();
    //
    //     merge_orientation_kernel.setArg(0, O_x_buffer);
    //     merge_orientation_kernel.setArg(1, O_y_buffer);
    //     merge_orientation_kernel.setArg(2, rotation_matrix_buffer);
    //
    //
    //     command_queue.enqueueNDRangeKernel(merge_orientation_kernel, cl::NullRange, cl::NDRange(1));
    //
    //     command_queue.finish();
    //
    //
    //     compute_descriptor_kernel.setArg(0, descriptor_buffer);
    //     compute_descriptor_kernel.setArg(1, image_buffer);
    //     compute_descriptor_kernel.setArg(2, patch_description_points_buffer);
    //     compute_descriptor_kernel.setArg(3, key_points_buffer);
    //     compute_descriptor_kernel.setArg(4, i); // communication hz
    //     compute_descriptor_kernel.setArg(5, rotation_matrix_buffer);
    //     compute_descriptor_kernel.setArg(6, test_cases_buffer);
    //     compute_descriptor_kernel.setArg(7, n_cols);
    //
    //     command_queue.enqueueNDRangeKernel(compute_descriptor_kernel, cl::NullRange, cl::NDRange(DESCRIPTOR_SIZE));
    //
    //     command_queue.finish();
    //     // break;
    //     // assert(err1 == 0 && err2 == 0 && err3 == 0);
    //     // if (err1 != CL_SUCCESS && err2 != CL_SUCCESS && err3 != CL_SUCCESS) {
    //     //     std::cerr << "Error" << std::endl;
    //     // }
    //     // command_queue.enqueueReadBuffer(descriptor_buffer, CL_TRUE, 0, sizeof(uint8_t) * flat_descriptor.size(), flat_descriptor.data());
    //     //
    //     // if (err4 != CL_SUCCESS) {
    //     //     std::cerr << "Error reading descriptor buffer! Error: " << err4 << "!" << std::endl;
    //     //     std::cerr << "Errored index: " << i << std::endl;
    //     // }
    // }
