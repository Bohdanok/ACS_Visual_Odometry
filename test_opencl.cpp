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
#include <opencv2/features2d.hpp>
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include "test_opencl.h"


// cl::Program create_platform(const std::string &filename) {
//     std::vector<cl::Platform> platforms;
//     cl::Platform::get(&platforms);
//
//     auto platform = platforms.front();
//     std::vector<cl::Device> devices;
//     platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
//
//     auto device = devices.front();
//
//     std::ifstream opened_file(filename);
//     std::string src((std::istreambuf_iterator<char>(opened_file)),
//                     (std::istreambuf_iterator<char>()));
//
//     cl::Program::Sources sources({{src.c_str(), src.length() + 1}});
//
//     cl::Context context(device);
//     cl::Program program(context, sources);
//
//     program.build("-cl-std=CL1.2");
//
//     return program;
//
// }

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



int main()
{
    const std::string kernel_path = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/kernels/gradient_convolution.cl";
    const std::string image_filename = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/images/Zhovkva2.jpg";
    const auto program = create_platform(kernel_path);

    const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    auto device = devices.front();
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();



    const cv::Mat img1 = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);
    cv::Mat my_blurred_gray;
    cv::GaussianBlur(img1, my_blurred_gray, cv::Size(7, 7), 0);


    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;

    cl::Buffer image_buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, n_cols * n_rows * sizeof(uchar), my_blurred_gray.data);

    cl::Buffer Jx_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));
    cl::Buffer Jy_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));
    cl::Buffer Jxy_buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, n_cols * n_rows * sizeof(double));


    cl::Kernel kernel(program, "gradient_convolution");

    kernel.setArg(0, image_buffer);
    kernel.setArg(1, Jx_buffer);
    kernel.setArg(2, Jy_buffer);
    kernel.setArg(3, Jxy_buffer);
    kernel.setArg(4, n_cols);


    cl::CommandQueue command_queue(context, device);

    cl::NDRange global_offset(1, 1);                       // Start at (1, 1)
    cl::NDRange global_size(n_cols - 2, n_rows - 2);       // Size = (n_cols - 2, n_rows - 2)

    command_queue.enqueueNDRangeKernel(kernel, global_offset, global_size);



    // cv::Mat Jx_mat = cv::Mat::zeros(n_rows, n_cols, CV_64F);
    // cv::Mat Jy_mat = cv::Mat::zeros(n_rows, n_cols, CV_64F);
    // cv::Mat Jxy_mat = cv::Mat::zeros(n_rows, n_cols, CV_64F);
    //
    // command_queue.enqueueReadBuffer(Jx_buffer, CL_TRUE, 0, sizeof(double) * n_rows * n_cols, reinterpret_cast<void*>(Jx_mat.ptr<double>()));
    // command_queue.enqueueReadBuffer(Jy_buffer, CL_TRUE, 0, sizeof(double) * n_rows * n_cols, reinterpret_cast<void*>(Jy_mat.ptr<double>()));
    // command_queue.enqueueReadBuffer(Jxy_buffer, CL_TRUE, 0, sizeof(double) * n_rows * n_cols, reinterpret_cast<void*>(Jxy_mat.ptr<double>()));
    //
    // cv::imshow("Jx", Jx_mat);
    // cv::imshow("Jy", Jy_mat);
    // cv::imshow("Jxy", Jxy_mat);
    //
    // cv::waitKey(0);
    // cv::destroyAllWindows();


}
