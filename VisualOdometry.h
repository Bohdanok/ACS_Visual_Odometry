#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include <cstddef>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "feature_extraction_parallel_GPU/feature_extraction_parallel_GPU.h"
#include "feature_matching_parallel/feature_matching_parallel.h"
#include "PoseUpdate.hpp"
#include "ransac.hpp"

class VisualOdometry {
private:
    thread_pool VO_pool;
    const std::size_t number_of_threads;
    GPU_settings gpu_settings;
    const std::string kernel_filename;

public:
    explicit VisualOdometry(const std::string& kernel_filename, std::size_t num_threads);
    // std::vector<std::vector<uint8_t>> compute_descriptor(cv::Mat& image);
    std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>>
        compute_descriptor_with_key_points(const cv::Mat& image);
    std::vector<std::pair<int, int>> match_descriptors(
        const std::vector<std::vector<uint8_t>>& desc1,
        const std::vector<std::vector<uint8_t>>& desc2);
    void run(const std::string image_dir, size_t num_images, const std::string pose_file, const std::string output_csv);
};

#endif // VISUAL_ODOMETRY_H