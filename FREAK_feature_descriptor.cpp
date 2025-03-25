//
// Created by julfy1 on 3/24/25.
//

#include "FREAK_feature_descriptor.h"
#include <iostream>
// inline static auto prepare_the_surroundings(const cv::Mat& blurred_gray_picture, const std::vector<int>& key_point, const int& n_cols, const int& n_rows) {
//             const std::vector<int> base_radii = {0, 2, 3, 5, 7, 10, 15}; // might be 1 to always blur the surroundings
//
//             const double scale_factor = std::min(n_rows, n_cols) / 100; //22.5
//
//             const double x_norm = static_cast<double>(key_point[1]) / static_cast<double>(n_cols);
//             const double y_norm = static_cast<double>(key_point[0]) / static_cast<double>(n_rows);
//
//
//             const double distance_form_the_origin = std::sqrt(std::pow(x_norm - 0.5, 2) + std::pow(y_norm - 0.5, 2));
//
//             const int index = std::min(static_cast<int>(distance_form_the_origin * base_radii.size()), static_cast<int>(base_radii.size() - 1));
//             const int radius = base_radii[index] * scale_factor;
//             if (radius == 0) {return;}
//             std::cout << "Point: <" << key_point[0] << ", " << key_point[1] << ">" << "\tNormed distance from the origin: " << distance_form_the_origin << "\tRadius: " << radius << std::endl;
//
//             const int x = std::max(0, key_point[1] - radius);
//             int const y = std::max(0, key_point[0] - radius);
//             int const width = std::min(n_cols - x, 2 * radius);
//             int const height = std::min(n_rows - y, 2 * radius);
//
//             const cv::Rect roi(x, y, width, height);
//
//             cv::Mat roi_image = blurred_gray_picture(roi);
//
//             const int odd_rad = (radius & 1) ? radius : radius + 1;
//
//             cv::GaussianBlur(roi_image, roi_image, cv::Size(odd_rad, odd_rad), odd_rad);
//
//         }

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

//
// void FREAK::FREAK_feature_description(const std::vector<std::tuple<int, int, double>>& key_points, cv::Mat blurred_gray_picture, const int& n_cols, const int& n_rows, const double corr_threshold) {
//
//     const size_t num_of_keypoints = key_points.size();
//     const size_t DESCRIPTOR_SIZE = 512;
//
//     // std::vector<std::pair<int, int>> predefined_point_for_matching = {
//     //     {33, 0}, {17, -30}, {-17, -30}, {-33, 0}, {-17, 30}, {17, 30},
//     //     {22, 13}, {22, -13}, {0, -26}, {-22, -13}, {-22, 13}, {0, 26},
//     //     {18, 0}, {9, -17}, {-9, -17}, {-18, 0}, {-9, 17}, {9, 17},
//     //     {11, 7}, {11, -7}, {0, -13}, {-11, -7}, {-11, 7}, {0, 13},
//     //     {8, 0}, {4, -8}, {-4, -8}, {-8, 0}, {-4, 8}, {4, 8},
//     //     {5, 3}, {5, -3}, {0, -6}, {-5, -3}, {-5, 3}, {0, 6},
//     //     {4, 0}, {2, -4}, {-2, -4}, {-4, 0}, {-2, 4}, {2, 4},
//     //     {0, 0}};
//
//
//     std::cout << "Num of cols: " << n_cols << std::endl;
//     std::cout << "Num of rows: " << n_rows << std::endl;
//
//     // std::vector<double> angles(key_points.size());
//
//
//     // In my case 903
//     // Here I consider the transpose of the matrix
//     std::vector<std::vector<double>> matrix_M(903, std::vector<double>(num_of_keypoints + 1, 0));
//     // std::vector<size_t> sums();
//     size_t count = 0;
//     for (size_t num = 0; num < num_of_keypoints; num++) {
//
//         const auto key_point = point(std::get<1>(key_points[num]), std::get<0>(key_points[num]));
//
//         // In my implementation I kick the points that are not in the bounds of retinal sampling pattern
//
//         // Rotate the image in the direction of the stronger intensity(for rotation invariance)
//         const double angle = compute_orientation(key_point , blurred_gray_picture, n_cols, n_rows);
//         const double rotation_matrix[4] = {std::cos(angle), -1 * std::sin(angle), std::sin(angle), std::cos(angle)};
//
//         size_t count = 0;
//         for (size_t i = 0; i < 43; i++) {
//             for (size_t j = i + 1; j < 43; j++) {
//                 // for (size_t key_point_index = 0; key_point_index < 903; key_point_index++) {
//
//                 const auto pt1 = predefined_point_for_matching[i];
//                 const auto pt2 = predefined_point_for_matching[j];
//
//                 const point pnt1 = point(key_point .x +
//                     std::get<0>(pt1) * rotation_matrix[0] + std::get<1>(pt1) * rotation_matrix[2],
//                     key_point .y + (-1) * std::get<0>(pt1) * rotation_matrix[1] + std::get<1>(pt1) * rotation_matrix[3]);
//
//                 const point pnt2 = point(key_point .x +
//                     std::get<0>(pt2) * rotation_matrix[0] + std::get<1>(pt2) * rotation_matrix[2],
//                     key_point .y + (-1) * std::get<0>(pt2) * rotation_matrix[1] + std::get<1>(pt2) * rotation_matrix[3]);
//
//                 const double add = blurred_gray_picture.at<double>(pnt1.y, pnt1.x) > blurred_gray_picture.at<double>(pnt2.y, pnt2.x) ? 1 : 0;
//                 // const double add = 1;
//                 matrix_M[count][num] = add;
//
//                 matrix_M[count][num_of_keypoints] += add;
//                 count++;
//             }
//
//         }
//     }
//     const double num_of_keypoints_double_div2 = num_of_keypoints / 2.0;
//     std::sort(matrix_M.begin(), matrix_M.end(), [num_of_keypoints_double_div2, num_of_keypoints](const std::vector<double> &a, const std::vector<double> &b) {return std::abs(a[num_of_keypoints] - num_of_keypoints_double_div2) < std::abs(b[num_of_keypoints] - num_of_keypoints_double_div2);});
//
//     // debug
//     //
//     // for (size_t i = 0; i < 903; i++) {
//     //     std::cout << matrix_M[i][num_of_keypoints] << ", ";
//     // }
//
//     // find the columns with the least correlation
//
//     std::vector<std::vector<int>> output_matrix(num_of_keypoints + 1, std::vector<int>(DESCRIPTOR_SIZE));
//     add_transposed_vector(output_matrix, matrix_M[0], 0, num_of_keypoints + 1);
//     matrix_M.erase(matrix_M.begin());
//     count = 1;
//     size_t column_count = 1;
//
//     while (column_count < DESCRIPTOR_SIZE) {
//         if (column_count == 106) {
//             std::cout << "Hi";
//         }
//         double lowest_correlation = std::numeric_limits<double>::max();
//         double correlation_sum = 0;
//         int min_index = -1;
//         for (size_t i = 0; i < matrix_M.size(); i++) {
//
//             // calc the standart deviation of cur point
//             // calc the columns, so good
//
//             const std::vector<double> cur_row = matrix_M[i];
//
//             double sum_input_set_square = 0;
//
//             const double sample_mean_input_set = cur_row[num_of_keypoints] / static_cast<double>(num_of_keypoints);
//             for (size_t n = 0; n < num_of_keypoints; n++) {
//                 sum_input_set_square += std::pow(cur_row[n] - sample_mean_input_set, 2);
//             }
//
//
//             for (size_t j = 0; j < column_count; j++) {
//
//                 double sum_output_set_square = 0;
//                 double sum_product = 0;
//                 const double sample_mean_output_set = output_matrix[num_of_keypoints][j] / static_cast<double>(num_of_keypoints);
//                 for (size_t n = 0; n < num_of_keypoints; n++) {
//                     sum_output_set_square += std::pow(output_matrix[column_count][n] - sample_mean_output_set, 2);
//                 }
//
//                 for (size_t n = 0; n < num_of_keypoints; n++) {
//                     sum_product += (cur_row[n] - sample_mean_input_set) * (output_matrix[column_count][n] - sample_mean_output_set);
//                 }
//
//                 const double curr_correlation = std::abs(sum_product / (std::sqrt(sum_input_set_square * sum_output_set_square)));
//                 if (curr_correlation > corr_threshold || std::isnan(curr_correlation)) {
//                     std::cout << "Skipped: " << std::endl;
//                     break;
//                 }
//                 // std::cout << "Correlation sum: " << correlation_sum << std::endl;
//                 correlation_sum += curr_correlation;
//             }
//
//             if (correlation_sum < lowest_correlation) {
//                 lowest_correlation = correlation_sum;
//                 std::cout << "Lowest correlation: " << lowest_correlation << std::endl;
//                 std::cout << "Iteration index: " << column_count << std::endl;
//                 min_index = i;
//             }
//         }
//
//         if (min_index == -1) {
//             std::cerr << ("There are no suitable columns to add to the final descriptor! The iteration <"
//                     + std::to_string(column_count) + ">! Correlation sum: " + std::to_string(correlation_sum)) << std::endl;
//         }
//
//         add_transposed_vector(output_matrix, matrix_M[min_index], column_count, num_of_keypoints);
//         matrix_M.erase(matrix_M.begin() + min_index);
//         column_count++;
//     }
//
//     std::cout << "Descriptor length: " << output_matrix[0].size() << std::endl;
//     std::cout << "Matrix length: " << output_matrix.size() << std::endl;
//
//
//     cv::imshow("BOhdan with description", blurred_gray_picture);
// }


std::vector<std::vector<uint8_t>> FREAK::FREAK_feature_description(const std::vector<point>& key_points, cv::Mat blurred_gray_picture, const int& n_cols, const int& n_rows, const double corr_threshold) {

    const size_t num_of_keypoints = key_points.size();
    std::vector<std::vector<uint8_t>> descriptor(num_of_keypoints, std::vector<uint8_t>(DESCRIPTOR_SIZE));
    std::cout << "Rows: " << n_rows << "\t" << "Cols: " << n_cols << std::endl;

    for (size_t i = 0; i < num_of_keypoints; i++) {

        const auto key_point = key_points[i];

        const double angle = compute_orientation(key_point , blurred_gray_picture, n_cols, n_rows);
        const double rotation_matrix[4] = {std::cos(angle), -1 * std::sin(angle), std::sin(angle), std::cos(angle)};

        for (size_t j = 0; j < DESCRIPTOR_SIZE; j++) {

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

            if (blurred_gray_picture.at<uchar>(pnt1.y, pnt1.x) > blurred_gray_picture.at<uchar>(pnt2.y, pnt2.x)) {
                descriptor[i][j] = 1;
            }
            else {
                descriptor[i][j] = 0;
            }
            // std::cout << "i: " << i << ", j: " << j << ", descriptor: " << int(descriptor[i][j]) << std::endl;
        }

    }

    return descriptor;

}
