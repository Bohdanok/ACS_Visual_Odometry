//
// Created by julfy1 on 3/24/25.
//

#include "FREAK_feature_descriptor_parallel.h"
#include <iostream>

float FREAK_Parallel::compute_orientation(const cv::KeyPoint &point, const cv::Mat& image) {

    float O_x = 0;
    float O_y = 0;

    for (int i = 0; i < predefined_point_for_matching.size(); i++) {
        const auto point1_intensity = static_cast<float>(image.at<uchar>(predefined_point_for_matching[i].y + point.pt.y, predefined_point_for_matching[i].x + point.pt.x));
        for (int j = i + 1; j < predefined_point_for_matching.size(); j++) {
            const float intensity_change = point1_intensity - static_cast<float>(image.at<uchar>(predefined_point_for_matching[j].y + point.pt.y, predefined_point_for_matching[j].x + point.pt.x));
            // norm of 2 vectors
            const float norm = sqrt(std::pow(predefined_point_for_matching[i].y + point.pt.y - predefined_point_for_matching[j].y + point.pt.y, 2) +
                std::pow(predefined_point_for_matching[i].x + point.pt.x - predefined_point_for_matching[j].x + point.pt.x, 2));
            O_x += intensity_change * (predefined_point_for_matching[i].x - predefined_point_for_matching[j].x) / norm;
            O_y += intensity_change * (predefined_point_for_matching[i].y - predefined_point_for_matching[j].y) / norm;
        }
    }
    if (std::isnan(O_x) || std::isnan(O_y)) { // a check for isinf() might be useful here
        return 0;
    }
    return std::atan2(O_y, O_x); // div by M?? Nah, I'd win

}


void FREAK_Parallel::FREAK_feature_description(const std::vector<cv::KeyPoint>& key_points, cv::Mat blurred_gray_picture, const size_t& starting_key_point_index, std::vector<std::vector<uint8_t>>& descriptor) {

    const size_t num_of_keypoints = key_points.size();
    // std::vector<std::vector<uint8_t>> descriptor(num_of_keypoints, std::vector<uint8_t>(DESCRIPTOR_SIZE));
    // std::cout << "Rows: " << n_rows << "\t" << "Cols: " << n_cols << std::endl;

    for (size_t i = 0; i < num_of_keypoints; i++) {

        const auto key_point = key_points[i];

        const float angle = compute_orientation(key_point , blurred_gray_picture);
        const float rotation_matrix[4] = {std::cos(angle), -1 * std::sin(angle), std::sin(angle), std::cos(angle)};

        for (size_t j = 0; j < DESCRIPTOR_SIZE; j++) {
            // std::cout << "Key point: " << "(" << key_point.x << ", " << key_point.y << ")" << std::endl;
            const auto cur_patch = test_cases[j];

            const auto pt1 = cur_patch.point1;
            const auto pt2 = cur_patch.point2;

            const point pnt1 = point( static_cast<int>(key_point.pt.x +
                pt1.pt.x * rotation_matrix[0] + pt1.pt.y * rotation_matrix[2]),
                static_cast<int>(key_point.pt.y + (-1) * pt1.pt.x * rotation_matrix[1] + pt1.pt.y * rotation_matrix[3]));

            const point pnt2 = point(static_cast<int>(key_point.pt.x +
                pt2.pt.x * rotation_matrix[0] + pt2.pt.y * rotation_matrix[2]),
                static_cast<int>(key_point.pt.y + (-1) * pt2.pt.x * rotation_matrix[1] + pt2.pt.y * rotation_matrix[3]));

            if (blurred_gray_picture.at<uchar>(pnt1.y, pnt1.x) > blurred_gray_picture.at<uchar>(pnt2.y, pnt2.x)) {
                descriptor[i][j] = 1;
            }
            else {
                descriptor[i][j] = 0;
            }
            // return descriptor; // debug

        }

    }

}



void FREAK_Parallel::FREAK_feature_description_worker(const std::vector<cv::KeyPoint>& key_points, const cv::Mat& blurred_gray_picture, const size_t& starting_key_point_index, std::vector<std::vector<uint8_t>>& descriptor, const size_t& num_of_keypoints, const size_t& KEYPOINTS_PER_TASK) {

    const size_t upper_bound = std::min(num_of_keypoints, starting_key_point_index + KEYPOINTS_PER_TASK);

    for (size_t i = starting_key_point_index; i < upper_bound; i++) {

        const auto key_point = key_points[i];

        const float angle = compute_orientation(key_point , blurred_gray_picture);
        const float rotation_matrix[4] = {std::cos(angle), -1 * std::sin(angle), std::sin(angle), std::cos(angle)};

        for (size_t j = 0; j < DESCRIPTOR_SIZE; j++) {
            // std::cout << "Key point: " << "(" << key_point.x << ", " << key_point.y << ")" << std::endl;
            const auto cur_patch = test_cases[j];

            const auto pt1 = cur_patch.point1;
            const auto pt2 = cur_patch.point2;

            const point pnt1 = point( static_cast<int>(key_point.pt.x +
                pt1.pt.x * rotation_matrix[0] + pt1.pt.y * rotation_matrix[2]),
                static_cast<int>(key_point.pt.y + (-1) * pt1.pt.x * rotation_matrix[1] + pt1.pt.y * rotation_matrix[3]));

            const point pnt2 = point(static_cast<int>(key_point.pt.x +
                pt2.pt.x * rotation_matrix[0] + pt2.pt.y * rotation_matrix[2]),
                static_cast<int>(key_point.pt.y + (-1) * pt2.pt.x * rotation_matrix[1] + pt2.pt.y * rotation_matrix[3]));

            if (blurred_gray_picture.at<uchar>(pnt1.y, pnt1.x) > blurred_gray_picture.at<uchar>(pnt2.y, pnt2.x)) {
                descriptor[i][j] = 1;
            }
            else {
                descriptor[i][j] = 0;
            }

        }

    }

}
