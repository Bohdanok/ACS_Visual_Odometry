//
// Created by julfy on 20.04.25.
//

#include "VisualOdometry.h"

VisualOdometry::VisualOdometry(const size_t number_of_threads) : VO_pool(number_of_threads), number_of_threads(number_of_threads) {
    // + SOFIA SOME CODE FOR VO IDK
}

std::vector<std::vector<uint8_t>> VisualOdometry::compute_descriptor(cv::Mat &image) {
    if (image.channels() != 1) image = CornerDetectionParallel::custom_bgr2gray(image);
    return feature_extraction_manager(image, VO_pool);
}

std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> VisualOdometry::compute_descriptor_with_key_points(cv::Mat &image) {
    if (image.channels() != 1) image = CornerDetectionParallel::custom_bgr2gray(image);
    return feature_extraction_manager_with_points(image, VO_pool);
}

std::vector<std::pair<int, int>> VisualOdometry::match_descriptors(const std::vector<std::vector<uint8_t>> &desc1, const std::vector<std::vector<uint8_t>> &desc2) {
    return matchCustomBinaryDescriptorsThreadPool_v2(desc1, desc2, VO_pool);
}


// int main() {
//     std::cout << "Hi from main!" << std::endl;
// }
