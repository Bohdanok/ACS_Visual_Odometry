//
// Created by julfy on 13.04.25.
//
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "test_opencl.h"

unsigned int get_best_opencl_platform() {
    // ...
}


int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    auto device = devices.front();

}