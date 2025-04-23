//
// Created by julfy on 13.04.25.
//

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <fstream>
#include <iostream>
#include <vector>

#include "test_opencl.h"

#include <oneapi/tbb/detail/_task.h>

cl::Program create_platform(const std::string &filename) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    auto device = devices.front();

    std::ifstream opened_file(filename);
    std::string src((std::istreambuf_iterator<char>(opened_file)),
                    (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources({{src.c_str(), src.length() + 1}});

    cl::Context context(device);
    cl::Program program(context, sources);

    program.build("-cl-std=CL1.2");

    return program;

}

int main()
{
    const std::string kernel_path = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/kernels/process_2d_array.cl";
    const auto program = create_platform(kernel_path);

    const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    auto device = devices.front();
    auto context = program.getInfo<CL_PROGRAM_CONTEXT>();

    const int num_rows = 3;
    const int num_cols = 2;

    std::array<std::array<int, num_cols>, num_rows> arr = {{{1, 1},
                                                {2, 2},
                                                {3, 3}}};


    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_cols * num_rows * sizeof(int), arr.data());

    cl::Kernel kernel(program, "process_2d_array");

    kernel.setArg(0, buffer);

    cl::CommandQueue command_queue(context, device);

    command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num_cols, num_rows));
    command_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(int) * num_rows * num_cols, arr.data());

    for (auto elem1 : arr) {
        for (auto elem : elem1) {
            std::cout << elem << ", ";
        }
    }

}
