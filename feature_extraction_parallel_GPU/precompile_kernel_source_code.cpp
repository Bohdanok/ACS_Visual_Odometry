//
// Created by julfy on 24.04.25.
//

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    const std::string filename = "../kernels/feature_extraction_kernel_functions.c";

    // Load kernel source
    std::ifstream opened_file(filename);
    if (!opened_file.is_open()) {
        std::cerr << "Failed to open kernel source file\n";
        return 1;
    }

    std::string src((std::istreambuf_iterator<char>(opened_file)), std::istreambuf_iterator<char>());
    cl::Program::Sources sources({{src.c_str(), src.length() + 1}});

    // Get platform and device
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    // if (platforms.empty()) {
    //     std::cerr << "No OpenCL platforms found.\n";
    //     return 1;
    // }
    //
    // std::vector<cl::Device> devices;
    // platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    // if (devices.empty()) {
    //     std::cerr << "No OpenCL GPU devices found.\n";
    //     return 1;
    // }
    //
    // cl::Context context(devices[0]);
    // cl::Program program(context, sources);
    // if (program.build({devices[0]}) != CL_SUCCESS) {
    //     std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << "\n";
    //     return 1;
    // }

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }

    std::cout << "Available OpenCL Platforms:\n";
    for (size_t i = 0; i < platforms.size(); ++i) {
        std::string platformName;
        platforms[i].getInfo(CL_PLATFORM_NAME, &platformName);
        std::cout << "  [" << i << "] " << platformName << "\n";
    }

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        std::cerr << "No OpenCL GPU devices found.\n";
        return 1;
    }

    std::cout << "\nAvailable GPU Devices on Platform 0:\n";
    for (size_t i = 0; i < devices.size(); ++i) {
        std::string deviceName;
        devices[i].getInfo(CL_DEVICE_NAME, &deviceName);
        std::cout << "  [" << i << "] " << deviceName << "\n";
    }

    cl::Device chosenDevice = devices[0];
    std::string chosenName;
    chosenDevice.getInfo(CL_DEVICE_NAME, &chosenName);

    std::cout << "\nChosen device: " << chosenName << "\n";

    cl::Context context(chosenDevice);
    cl::Program program(context, sources);
    if (program.build({chosenDevice}) != CL_SUCCESS) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(chosenDevice) << "\n";
        return 1;
    }

    // Get binary sizes
    std::vector<size_t> binary_sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    std::vector<std::vector<unsigned char>> binaries = program.getInfo<CL_PROGRAM_BINARIES>();

    // Save first binary to file
    std::ofstream out("../kernels/feature_extraction_kernel_functions.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(binaries[0].data()), binary_sizes[0]);
    out.close();

    std::cout << "Kernel compiled and binary saved to feature_extraction_kernel_functions.bin\n";
    return 0;
}
