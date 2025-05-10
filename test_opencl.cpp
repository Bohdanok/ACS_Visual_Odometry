//
// Created by julfy on 13.04.25.
//

// int main()
// {
//     const std::string kernel_path = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/kernels/process_1d_array.cl";
//     const auto program = create_platform(kernel_path);
//
//     const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
//     auto device = devices.front();
//     auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
//
//     std::vector<int> input_data(1024);
//
//     // std::fill(input_data.begin(), input_data.end(), 1);
//
//     cl::Buffer in_buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * input_data.size(), input_data.data());
//     cl::Buffer out_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(int) * input_data.size());
//
//     cl::Kernel kernel(program, "process_1d_array");
//     kernel.setArg(0, in_buffer);
//     kernel.setArg(1, out_buffer);
//
//     cl::CommandQueue command_queue(context, device);
//
//     command_queue.enqueueFillBuffer(in_buffer, 3, sizeof(int) * 10, input_data.size() * 10 - sizeof(int) * 10);
//
//     command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_data.size()));
//
//     command_queue.enqueueReadBuffer(out_buffer, CL_FALSE, 0, sizeof(int) * input_data.size(), input_data.data());
//
//     cl::finish();
//
//     std::cout << "Hi!" << std::endl;
//
// }

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#define CL_HPP_ENABLE_EXCEPTIONS

#include "test_opencl.h"

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


cl::Program create_platform(const std::string &filename) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();

    std::ifstream opened_file(filename);
    std::string src((std::istreambuf_iterator<char>(opened_file)), std::istreambuf_iterator<char>());
    cl::Program::Sources sources({{src.c_str(), src.length() + 1}});

    cl::Context context(device);
    cl::Program program(context, sources);

    try {
        program.build("-cl-std=CL1.2");
    } catch (const ::cl::Error& err)
    {
        for (const auto& d : devices) {
            std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d);
            std::cerr << "Build Log for device " << d.getInfo<CL_DEVICE_NAME>() << ":\n";
            std::cerr << buildLog << "\n";
        }
        throw;
    }

    return program;
}

cl::Program create_platform_from_binary(const std::string &binary_filename) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();

    std::ifstream bin_file(binary_filename, std::ios::binary | std::ios::ate);
    if (!bin_file.is_open()) {
        throw std::runtime_error("Failed to open kernel binary file: " + binary_filename);
    }

    std::streamsize size = bin_file.tellg();
    bin_file.seekg(0, std::ios::beg);
    std::vector<unsigned char> binary(size);
    if (!bin_file.read(reinterpret_cast<char*>(binary.data()), size)) {
        throw std::runtime_error("Failed to read kernel binary file: " + binary_filename);
    }

    cl::Context context(device);
    cl::Program::Binaries binaries = { binary };
    cl::Program program(context, {device}, binaries);

    try {
        program.build({device});
    } catch (const cl::Error& err) {
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Build Log for device " << device.getInfo<CL_DEVICE_NAME>() << ":\n";
        std::cerr << buildLog << "\n";
        throw;
    }

    return program;
}



int main()
{

    const std::string kernel_path = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/kernels/feature_extraction_kernel_functions.bin";

    const std::string image_filename = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/images/Zhovkva2.jpg";

    const auto program = create_platform_from_binary(kernel_path);

    const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    const auto& device = devices.front();
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();

    // std::string version = device.getInfo<CL_DEVICE_VERSION>();
    // std::cout << "Device version: " << version << std::endl;

    const cv::Mat img1 = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);
    auto start = get_current_time_fenced();

    cv::Mat my_blurred_gray;
    cv::GaussianBlur(img1, my_blurred_gray, cv::Size(7, 7), 0);


    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;

    cl::Buffer image_buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, n_cols * n_rows * sizeof(uchar), my_blurred_gray.data);

    cl::Buffer Jx_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));
    cl::Buffer Jy_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));
    cl::Buffer Jxy_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));


    cl::Kernel kernel_gradient(program, "gradient_convolution");

    kernel_gradient.setArg(0, image_buffer);
    kernel_gradient.setArg(1, Jx_buffer);
    kernel_gradient.setArg(2, Jy_buffer);
    kernel_gradient.setArg(3, Jxy_buffer);
    kernel_gradient.setArg(4, n_cols);


    const cl::CommandQueue command_queue(context, device);

    const cl::NDRange global_offset_gradient(1, 1);
    const cl::NDRange global_size_gradient(n_cols - 2, n_rows - 2);

    command_queue.enqueueNDRangeKernel(kernel_gradient, global_offset_gradient, global_size_gradient);

    command_queue.finish();

    // R score

    cl::Kernel kernel_shitomasi_response(program, "shitomasi_response");

    cl::Buffer R_response_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));


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

    // std::vector<std::vector<double>> R_array(n_rows, std::vector<double>(n_cols, 0));
    //
    // command_queue.enqueueReadBuffer(R_response_buffer, CL_TRUE, 0, sizeof(double) * n_rows * n_cols, R_array.data());

    std::vector<double> flat_R_array(n_rows * n_cols);

    command_queue.enqueueReadBuffer(R_response_buffer, CL_TRUE, 0, sizeof(double) * flat_R_array.size(), flat_R_array.data());

    std::vector<std::vector<double>> R_array(n_rows, std::vector<double>(n_cols));
    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_cols; ++j)
            R_array[i][j] = flat_R_array[i * n_cols + j];


    auto end = get_current_time_fenced();
    std::cout << "GPU gradient calculations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    // draw_score_distribution(R_array, "Shi-Tomasi response");
    // cv::waitKey(0);
    // cv::destroyAllWindows();



}
