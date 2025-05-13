//
// Created by julfy on 24.04.25.
//

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    const std::string filename = "../kernels/feature_extraction_kernel_functions.c";

    std::ifstream opened_file(filename);
    if (!opened_file.is_open()) {
        std::cerr << "Failed to open kernel source file\n";
        return 1;
    }

    std::string src((std::istreambuf_iterator<char>(opened_file)), std::istreambuf_iterator<char>());
    cl::Program::Sources sources({{src.c_str(), src.length() + 1}});

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

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


    std::cout << "Number of OpenCL platforms: " << platforms.size() << "\n";

    for (size_t i = 0; i < platforms.size(); ++i) {
        std::string platformName, platformVendor, platformVersion;
        platforms[i].getInfo(CL_PLATFORM_NAME, &platformName);
        platforms[i].getInfo(CL_PLATFORM_VENDOR, &platformVendor);
        platforms[i].getInfo(CL_PLATFORM_VERSION, &platformVersion);

        std::cout << "\nPlatform [" << i << "]:\n";
        std::cout << "  Name    : " << platformName << "\n";
        std::cout << "  Vendor  : " << platformVendor << "\n";
        std::cout << "  Version : " << platformVersion << "\n";

        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        std::cout << "  Number of devices: " << devices.size() << "\n";

        for (size_t j = 0; j < devices.size(); ++j) {
            std::string deviceName, deviceVendor, deviceVersion;
            cl_device_type deviceType;
            cl_uint computeUnits;
            cl_ulong globalMemSize, localMemSize;
            size_t maxWorkGroupSize;
            std::vector<size_t> maxWorkItemSizes;

            devices[j].getInfo(CL_DEVICE_NAME, &deviceName);
            devices[j].getInfo(CL_DEVICE_VENDOR, &deviceVendor);
            devices[j].getInfo(CL_DEVICE_VERSION, &deviceVersion);
            devices[j].getInfo(CL_DEVICE_TYPE, &deviceType);
            devices[j].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &computeUnits);
            devices[j].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);
            devices[j].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
            devices[j].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
            devices[j].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &maxWorkItemSizes);

            std::cout << "  Device [" << j << "]:\n";
            std::cout << "    Name         : " << deviceName << "\n";
            std::cout << "    Vendor       : " << deviceVendor << "\n";
            std::cout << "    Version      : " << deviceVersion << "\n";
            std::cout << "    Type         : "
                      << ((deviceType & CL_DEVICE_TYPE_CPU) ? "CPU " : "")
                      << ((deviceType & CL_DEVICE_TYPE_GPU) ? "GPU " : "")
                      << ((deviceType & CL_DEVICE_TYPE_ACCELERATOR) ? "Accelerator " : "")
                      << "\n";
            std::cout << "    Compute Units: " << computeUnits << "\n";
            std::cout << "    Global Memory: " << globalMemSize / (1024 * 1024) << " MB\n";
            std::cout << "    Local Memory : " << localMemSize / 1024 << " KB\n";
            std::cout << "    Max Work Group Size: " << maxWorkGroupSize << "\n";
            std::cout << "    Max Work Item Sizes: ";
            for (size_t k = 0; k < maxWorkItemSizes.size(); ++k) {
                std::cout << maxWorkItemSizes[k];
                if (k < maxWorkItemSizes.size() - 1) std::cout << " x ";
            }
            std::cout << "\n";
        }
    }

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        std::cerr << "No OpenCL GPU devices found.\n";
        return 1;
    }

    std::cout << "\nAvailable GPU Devices on the chosen Platform:\n";
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
    try {
        program.build({chosenDevice}, "-cl-std=CL1.2");
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Build Error: " << err.what() << " (" << err.err() << ")\n";
        for (const auto& d : devices) {
            std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d);
            std::cerr << "Build Log for device " << d.getInfo<CL_DEVICE_NAME>() << ":\n";
            std::cerr << buildLog << "\n";
        }
        return 1;
    }


    std::vector<size_t> binary_sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    std::vector<std::vector<unsigned char>> binaries = program.getInfo<CL_PROGRAM_BINARIES>();

    std::ofstream out("../kernels/feature_extraction_kernel_functions.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(binaries[0].data()), binary_sizes[0]);
    out.close();

    std::cout << "Kernel compiled and binary saved to feature_extraction_kernel_functions.bin\n";
    return 0;
}
