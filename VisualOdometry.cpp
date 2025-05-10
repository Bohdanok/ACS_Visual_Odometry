//
// Created by julfy on 20.04.25.
//

#include "VisualOdometry.h"

VisualOdometry::VisualOdometry(
    const std::string& kernel_filename,
    const std::size_t num_threads)
  : VO_pool(num_threads)
  , number_of_threads(num_threads)
  , gpu_settings([&]() {
      // load binary and create OpenCL program
      const auto program = create_platform_from_binary(kernel_filename);
      const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
      if (devices.empty()) {
          throw std::runtime_error("No OpenCL devices found for the program.");
      }
      const auto& device = devices.front();
      const auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
      return GPU_settings{program, device, context};
  }())
  , kernel_filename(kernel_filename)
{   }


// READ in grayscale already
std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> VisualOdometry::compute_descriptor_with_key_points(const cv::Mat &image) {
    return feature_extraction_manager_with_points(image, this->gpu_settings);
}


std::vector<std::pair<int, int>> VisualOdometry::match_descriptors(const std::vector<std::vector<uint8_t>> &desc1, const std::vector<std::vector<uint8_t>> &desc2) {
    return matchCustomBinaryDescriptorsThreadPool(desc1, desc2, this->VO_pool, this->number_of_threads, 0.75f);
}

