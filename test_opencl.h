//
// Created by julfy on 13.04.25.
//

#ifndef TEST_OPENCL_H
#define TEST_OPENCL_H
#include <CL/cl2.hpp>


unsigned int get_best_opencl_platform();

cl::Program create_platform(const std::string &filename);

#endif //TEST_OPENCL_H
