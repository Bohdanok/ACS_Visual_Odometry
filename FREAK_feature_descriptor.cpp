//
// Created by julfy1 on 3/24/25.
//

#include "FREAK_feature_descriptor.h"
#include <iostream>

double FREAK::compute_orientation(point point, cv::Mat& image, const int& n_cols, const int& n_rows) {

    double O_x = 0;
    double O_y = 0;

    for (int i = 0; i < predefined_point_for_matching.size(); i++) {
        const auto point1_intensity = static_cast<double>(image.at<uchar>(predefined_point_for_matching[i].y + point.y, predefined_point_for_matching[i].x + point.x));
        for (int j = i + 1; j < predefined_point_for_matching.size(); j++) {
            const double intensity_change = point1_intensity - static_cast<double>(image.at<uchar>(predefined_point_for_matching[j].y + point.y, predefined_point_for_matching[j].x + point.x));
            // norm of 2 vectors
            const double norm = sqrt(std::pow(predefined_point_for_matching[i].y + point.y - predefined_point_for_matching[j].y + point.y, 2) +
                std::pow(predefined_point_for_matching[i].x + point.x - predefined_point_for_matching[j].x + point.x, 2));
            O_x += intensity_change * (predefined_point_for_matching[i].x - predefined_point_for_matching[j].x) / norm;
            O_y += intensity_change * (predefined_point_for_matching[i].y - predefined_point_for_matching[j].y) / norm;
        }
    }
    if (std::isnan(O_x) || std::isnan(O_y)) { // a check for isinf() might be useful here
        return 0;
    }
    return std::atan2(O_y, O_x); // div by M?? Nah, I'd win

}

void FREAK::add_transposed_vector(std::vector<std::vector<int>>& array, const std::vector<double>& add_vector, const size_t index, const size_t num_of_keypoints) {

    for (size_t i = 0; i < num_of_keypoints; i++) {

        array[i][index] = static_cast<int>(add_vector[i]);
        // std::cout << "Array: " << array[i][index] << std::endl;

    }
    std::cout << "Transposed something" << std::endl;

}

std::vector<std::vector<uint8_t>> FREAK::FREAK_feature_description(const std::vector<point>& key_points, cv::Mat blurred_gray_picture, const int& n_cols, const int& n_rows, const double corr_threshold) {

    const size_t num_of_keypoints = key_points.size();
    std::vector<std::vector<uint8_t>> descriptor(num_of_keypoints, std::vector<uint8_t>(DESCRIPTOR_SIZE));
    std::cout << "Rows: " << n_rows << "\t" << "Cols: " << n_cols << std::endl;

    for (size_t i = 0; i < num_of_keypoints; i++) {

        const auto key_point = key_points[i];

        const double angle = compute_orientation(key_point , blurred_gray_picture, n_cols, n_rows);
        const double rotation_matrix[4] = {std::cos(angle), -1 * std::sin(angle), std::sin(angle), std::cos(angle)};

        for (size_t j = 0; j < DESCRIPTOR_SIZE; j++) {
            // std::cout << "Key point: " << "(" << key_point.x << ", " << key_point.y << ")" << std::endl;
            const auto cur_patch = test_cases[j];

            const auto pt1 = cur_patch.point1;
            const auto pt2 = cur_patch.point2;

            const point pnt1 = point( static_cast<int>(key_point.x +
                pt1.x * rotation_matrix[0] + pt1.y * rotation_matrix[2]),
                static_cast<int>(key_point .y + (-1) * pt1.x * rotation_matrix[1] + pt1.y * rotation_matrix[3]));

            const point pnt2 = point(static_cast<int>(key_point .x +
                pt2.x * rotation_matrix[0] + pt2.y * rotation_matrix[2]),
                static_cast<int>(key_point .y + (-1) * pt2.x * rotation_matrix[1] + pt2.y * rotation_matrix[3]));
            // std::cout << "i: " << i << "\t" << "j: " << j << "\tPoint1: " << "(" << pnt1.x << ", " << pnt1.y << ")" << "\t" << "Point2: " << "(" << pnt2.x << ", " << pnt2.y << ")" << std::endl;

            // descriptor[i][j] = 1;

            // debug
            // assert(pnt1.y > n_rows)
            if (pnt1.y > n_rows || pnt1.x > n_cols) {
                std::cout << "Key point with out of bounds: " << "(" << key_point.x << ", " << key_point.y << ")" << std::endl;
                std::cout << "Before the rotation on " << angle << " radiants" << "\ti: " << i << "\t" << "j: " << j << "\tPoint1: " << "(" << pt1.x << ", " << pt1.y << ")" << std::endl;
                std::cout << "After the rotation on " << angle << " radiants" << "\ti: " << i << "\t" << "j: " << j << "\tPoint1: " << "(" << pnt1.x << ", " << pnt1.y << ")" << std::endl;
            }
            if (pnt2.y > n_rows || pnt2.x > n_cols) {
                std::cout << "Key point with out of bounds: " << "(" << key_point.x << ", " << key_point.y << ")" << std::endl;
                std::cout << "Before the rotation on " << angle << " radiants" << "\ti: " << i << "\t" << "j: " << j << "\tPoint2: " << "(" << pt2.x << ", " << pt2.y << ")" << std::endl;
                std::cout << "After the rotation on " << angle << " radiants" << "\ti: " << i << "\t" << "j: " << j << "\tPoint2: " << "(" << pnt2.x << ", " << pnt2.y << ")" << std::endl;
                // std::cout << "i: " << i << "\t" << "j: " << j << "\tPoint1: " << "(" << pnt1.x << ", " << pnt1.y << ")" << "\t" << "Point2: " << "(" << pnt2.x << ", " << pnt2.y << ")" << std::endl;
            }

            if (blurred_gray_picture.at<uchar>(pnt1.y, pnt1.x) > blurred_gray_picture.at<uchar>(pnt2.y, pnt2.x)) {
                descriptor[i][j] = 1;
            }
            else {
                descriptor[i][j] = 0;
            }
            // return descriptor; // debug
            // std::cout << "i: " << i << ", j: " << j << ", descriptor: " << int(descriptor[i][j]) << std::endl;
        }

    }

    return descriptor;

}
