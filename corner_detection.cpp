//
// Created by julfy1 on 2/1/25.
//

#include "corner_detection.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <pstl/utils.h>

cv::Mat CornerDetection::custom_bgr2gray(cv::Mat& picture) {
    const int n_rows = picture.rows;
    const int n_cols = picture.cols * 3;
    uchar* ptr_src;
    uchar* ptr_dst;
    cv::Mat output_picture = cv::Mat::zeros(n_rows, n_cols / 3, CV_8UC1);

    for (size_t i = 0; i < n_rows; i++) {
        ptr_src = picture.ptr<uchar>(i);
        ptr_dst = output_picture.ptr<uchar>(i);
        for (size_t j = 0; j < n_cols; j += 3) {
            ptr_dst[j / 3] = cv::saturate_cast<uchar>(0.114 * ptr_src[j] + 0.587 * ptr_src[j + 1] + 0.299 * ptr_src[j + 2]);
        }
    }
    return output_picture;

}

cv::Mat CornerDetection::test_their_sobel(const std::string& filename, const std::string& cur_path) { // Not needed, but there IDK
    cv::Mat image = cv::imread(cur_path + filename);

    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size(7, 7), 0);
    cv::Mat my_blurred_gray = CornerDetection::custom_bgr2gray(blurred);
    cv::Mat my_gray_regular = CornerDetection::custom_bgr2gray(image);

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;

    cv::Mat test_their_sobel = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);

    uchar* ptr_src1;
    uchar* ptr_src2;
    uchar* ptr_src3;
    uchar* ptr_dstx;
    uchar* ptr_dsty;

    constexpr int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    constexpr int gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };
    double sumx, sumy;
    for (int i = 1; i < n_rows - 1; i++) {
        for (int j = 1; j < n_cols - 1; j++) {
            sumx = 0.0;
            sumy = 0.0;
            // ptr_src1 = my_blurred_gray.ptr<uchar>(i + 1);

            for (int p = -1; p <= 1; p++) {
                for (int q = -1; q <= 1; q++) {
                    sumx += (my_blurred_gray.ptr<uchar>(i + p)[j + q] * gx[p + 1][q + 1]);
                    sumy += (my_blurred_gray.ptr<uchar>(i + p)[j + q] * gy[p + 1][q + 1]);
                }
            }
            test_their_sobel.ptr<uchar>(i)[j] = (uchar)sqrt(sumx * sumx + sumy * sumy);
        }
    }

    cv::imshow("THEIR TEST SOBEL", test_their_sobel);
    return test_their_sobel;

}
//
// void CornerDetection::compare_images(const cv::Mat& image_my, const cv::Mat& image_their, const std::string win_name) {
//
//     cv::Mat output_check, return_something;
//
//     // cv::bitwise_xor(image_my, image_their, output_check);
//     cv::subtract(image_their, image_my, return_something);
//
//     cv::imshow(win_name, return_something);
// }

std::vector<cv::Mat> CornerDetection::direction_gradients(cv::Mat& picture, const int& n_rows, const int& n_cols) {

    uchar* ptr_src1;
    uchar* ptr_src2;
    uchar* ptr_src3;
    double* ptr_dst_Jx;
    double* ptr_dst_Jy;
    double sumx[3] = {0};
    double sumy[3] = {0};
    // cv::Mat test_sobel = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);

    cv::Mat Jx = cv::Mat::zeros(n_rows, n_cols, CV_64F);
    cv::Mat Jy = cv::Mat::zeros(n_rows, n_cols, CV_64F);
    cv::Mat Jxy = cv::Mat::zeros(n_rows, n_cols, CV_64F);

    double* ptr_dst_Jxy;

    for (size_t i = 1; i < n_rows - 1; i++) {

        ptr_src1 = picture.ptr<uchar>(i - 1);
        ptr_src2 = picture.ptr<uchar>(i);
        ptr_src3 = picture.ptr<uchar>(i + 1);
        ptr_dst_Jx = Jx.ptr<double>(i);
        ptr_dst_Jy = Jy.ptr<double>(i);
        ptr_dst_Jxy = Jxy.ptr<double>(i);

        for (size_t j = 1; j < n_cols - 1; j++) {

            for (short k = -1; k <= 1; k++) {
                sumx[k + 1] = ptr_src1[j + k] - ptr_src3[j + k]; // [1, 0, -1] T
                sumy[k + 1] = ptr_src1[j + k] + 2 * ptr_src2[j + k] + ptr_src3[j + k]; // [1, 2, 1] T

            }

            ptr_dst_Jx[j] = sumx[0] + 2 * sumx[1] + sumx[2]; // [1, 2, 1]
            ptr_dst_Jy[j] = sumy[0] - sumy[2]; // [1, 0, -1]
            ptr_dst_Jxy[j] = sumx[0] - sumx[2]; // [1, 0, -1]

        }
    }

    std::vector<cv::Mat> output;
    output.push_back(Jx);
    output.push_back(Jy);
    output.push_back(Jxy);
    return output;
}

cv::Mat CornerDetection::sobel_filter(cv::Mat& Jx, cv::Mat& Jy, const int& n_rows, const int& n_cols) {

    cv::Mat sobel_filtered = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);

    double* ptr_srcjx;
    double* ptr_srcjy;
    uchar* ptr_dst;
    for (size_t i = 0; i < n_rows; i++) {

        ptr_srcjx = Jx.ptr<double>(i);
        ptr_srcjy = Jy.ptr<double>(i);
        ptr_dst = sobel_filtered.ptr<uchar>(i);

        for (size_t j = 0; j < n_cols; j++) {
            ptr_dst[j] = (uchar)(sqrt(ptr_srcjx[j] * ptr_srcjx[j] + ptr_srcjy[j] * ptr_srcjy[j]));
        }
    }
    return sobel_filtered;
}

std::vector<std::vector<double>> CornerDetection::harris_corner_detection(cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, const int& n_rows, const int& n_cols, const double& k) {

    double jx2, jy2, det, trace, R;
    // const double k = 0.07;
    double max_R = -DBL_MIN; // I've seen somewhere the implementation of thresholding with threshold = 0.01 * max(R)

    double* ptr_srcjx;
    double* ptr_srcjy;
    double* ptr_srcjxy;

    double sumjxy;

    std::vector<std::vector<double>> R_array(n_rows, std::vector<double>(n_cols, 0));

    for (int i = 2; i < n_rows - 2; i++) {

        for (int j = 2; j < n_cols - 2; j++) {

            // R = det(M)−k⋅(trace(M))**2
            sumjxy = 0;
            jx2 = 0, jy2 = 0;

            for (int m = -2; m <= 2; m++) {
                ptr_srcjx = Jx.ptr<double>(i + m);
                ptr_srcjy = Jy.ptr<double>(i + m);
                ptr_srcjxy = Jxy.ptr<double>(i + m);

                for (int n = -2; n <= 2; n++) {
                    double jx = ptr_srcjx[j + n];
                    double jy = ptr_srcjy[j + n];
                    double jxy = ptr_srcjxy[j + n];

                    // sumjx += jx;
                    // sumjy += jy;
                    sumjxy += jxy;

                    jx2 += jx * jx;  // Accumulate squared Jx values
                    jy2 += jy * jy;  // Accumulate squared Jy values
                }
            }

            det = (jx2 * jy2) - (sumjxy * sumjxy);
            trace = jx2 + jy2;

            R = det - (k * trace * trace);
            // R = std::min(jx2, jy2);

            // std::cout << "R: " << R << std::endl;
            max_R = std::max(max_R, R);

            R_array[i][j] = R;

        }

    }

    //Ba bemba
    const double threshold = max_R * 0.01;
    for (int i = 2; i < n_rows - 2; i++) {
        for (int j = 2; j < n_cols - 2; j++) {

            if (R_array[i][j] <= threshold) {
                    // std::cout << "R > " << threshold << ": (" << i << ", " << j << ")" << std::endl;
                    R_array[i][j] = 0;
            }
        }
    }
    return R_array;
}

std::vector<std::vector<double>> CornerDetection::shitomasi_corner_detection(cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, const int& n_rows, const int& n_cols, const double& k) {


    double jx2, jy2, det, trace, R;
    // const double k = 0.07;
    double max_R = -DBL_MIN; // I've seen somewhere the implementation of thresholding with threshold = 0.01 * max(R)

    double* ptr_srcjx;
    double* ptr_srcjy;
    double* ptr_srcjxy;

    double sumjxy;

    std::vector<std::vector<double>> R_array(n_rows, std::vector<double>(n_cols, 0));

    for (int i = 2; i < n_rows - 2; i++) {

        for (int j = 2; j < n_cols - 2; j++) {

            // R = det(M)−k⋅(trace(M))**2
            sumjxy = 0;
            jx2 = 0, jy2 = 0;

            for (int m = -2; m <= 2; m++) {
                ptr_srcjx = Jx.ptr<double>(i + m);
                ptr_srcjy = Jy.ptr<double>(i + m);
                ptr_srcjxy = Jxy.ptr<double>(i + m);

                for (int n = -2; n <= 2; n++) {
                    double jx = ptr_srcjx[j + n];
                    double jy = ptr_srcjy[j + n];
                    double jxy = ptr_srcjxy[j + n];

                    // sumjx += jx;
                    // sumjy += jy;
                    sumjxy += jxy;

                    jx2 += jx * jx;  // Accumulate squared Jx values
                    jy2 += jy * jy;  // Accumulate squared Jy values
                }
            }

            det = (jx2 * jy2) - (sumjxy * sumjxy);
            trace = jx2 + jy2;

            R = (trace / 2) - (0.5 * std::sqrt(trace * trace - 4 * det));
            // R = std::min(jx2, jy2);

            // std::cout << "R: " << R << std::endl;
            max_R = std::max(max_R, R);

            R_array[i][j] = R;

        }

    }

    //Ba bemba
    const double threshold = max_R * 0.01;
    for (int i = 2; i < n_rows - 2; i++) {
        for (int j = 2; j < n_cols - 2; j++) {

            if (R_array[i][j] <= threshold) {
                    // std::cout << "R > " << threshold << ": (" << i << ", " << j << ")" << std::endl;
                    R_array[i][j] = 0;
            }

        }
    }
    return R_array;

}


        // void CornerDetection::draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name) {
        //
        //     int rows = R_values.size();
        //     int cols = R_values[0].size();
        //
        //     cv::Mat mat(rows, cols, CV_64F); // Create matrix to store values
        //
        //     // Copy values
        //     for (int i = 0; i < rows; ++i) {
        //         for (int j = 0; j < cols; ++j) {
        //             mat.at<double>(i, j) = R_values[i][j];
        //         }
        //     }
        //
        //     // Normalize values to range [0, 1]
        //     double minVal, maxVal;
        //     cv::minMaxLoc(mat, &minVal, &maxVal);
        //     cv::Mat normMat = (mat - minVal) / (maxVal - minVal); // Normalize between 0-1
        //
        //     // Create color image
        //     cv::Mat colorImage(rows, cols, CV_8UC3);
        //
        //     for (int i = 0; i < rows; ++i) {
        //         for (int j = 0; j < cols; ++j) {
        //             double val = normMat.at<double>(i, j); // Normalized value [0, 1]
        //
        //             // Map to RGB colors (Green → Blue → Red)
        //             uchar red   = static_cast<uchar>(255 * std::max(0.0, (val - 0.5) * 2));  // Increase red for higher values
        //             uchar blue  = static_cast<uchar>(255 * std::max(0.0, (0.5 - std::abs(val - 0.5)) * 2));  // Max in the middle
        //             uchar green = static_cast<uchar>(255 * std::max(0.0, (0.5 - val) * 2));  // Decrease green as value increases
        //
        //             colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        //         }
        //     }
        //     cv::imshow(win_name, colorImage);
        //     cv::imwrite("../test_images/output_images/" + win_name + ".png", colorImage);
        //
        // }


std::vector<point> CornerDetection::non_maximum_suppression(std::vector<std::vector<double>> R_values, const int& n_rows, const int& n_cols, const int& k, const int& N) {
    std::priority_queue<std::tuple<double, int, int>> max_heap; // Store (R_value, i, j)
    std::vector<point> output_corners;
    output_corners.reserve(N);
    int count = 0;

    for (int i = k / 2; i < n_rows - k / 2; i++) {
        for (int j = k / 2; j < n_cols - k / 2; j++) {

            // not to include out of bounce for retinal sampling

            if (!((j >= 37) && (j <= n_cols - 37) && (i >= 35) && (i <= n_rows - 35))) {
                continue;
            }

            double center_val = R_values[i][j];
            bool is_local_max = true;

            for (int n = i - k / 2; n <= i + k / 2; n++) {
                for (int m = j - k / 2; m <= j + k / 2; m++) {
                    if (!(n == i && m == j)) {
                        if (R_values[n][m] >= center_val) {
                            is_local_max = false;
                            break;
                        }
                    }
                }
                if (!is_local_max) break;
            }

            if (is_local_max) {
                max_heap.push({center_val, i, j});
            }
        }
    }

    for (int i = 0; i < N && !max_heap.empty(); i++) {
        output_corners.push_back({std::get<2>(max_heap.top()), std::get<1>(max_heap.top())});
        max_heap.pop();
        count++;
    }
    std::cout << "COunt: " << count << std::endl; // debug
    return output_corners;
}


        // const double largets_radius = 2 / 3;
        //
        // const double smallest_radius = 2 / 24;
        //
        // const double retinal_spacing = (largets_radius - smallest_radius) / 21;
        //
        // const std::vector<double> retinal_keypoint_radii = {largets_radius, largets_radius - 6 * smallest_radius, largets_radius - 11 * smallest_radius,
        //                  largets_radius - 15 * smallest_radius, largets_radius - 18 * smallest_radius,
        //                  largets_radius - 20 * smallest_radius, smallest_radius, 0};
        //
        // const std::vector<double> retinal_distances_from_the_center = {largets_radius / 2.0, (largets_radius - 6 * smallest_radius) / 2.0, (largets_radius - 11 * smallest_radius) / 2.0,
        //              (largets_radius - 15 * smallest_radius) / 2.0, (largets_radius - 18 * smallest_radius) / 2.0,
        //              (largets_radius - 20 * smallest_radius) / 2.0, smallest_radius / 2.0, 0};
        //////////////////////////////////////////////////////////////////////////////////////FREAKS

        // inline static auto prepare_the_surroundings(const cv::Mat& blurred_gray_picture, const std::vector<int>& key_point, const int& n_cols, const int& n_rows) {
        //     const std::vector<int> base_radii = {0, 2, 3, 5, 7, 10, 15}; // might be 1 to always blur the surroundings
        //
        //     const double scale_factor = std::min(n_rows, n_cols) / 100; //22.5
        //
        //     const double x_norm = static_cast<double>(key_point[1]) / static_cast<double>(n_cols);
        //     const double y_norm = static_cast<double>(key_point[0]) / static_cast<double>(n_rows);
        //
        //
        //     const double distance_form_the_origin = std::sqrt(std::pow(x_norm - 0.5, 2) + std::pow(y_norm - 0.5, 2));
        //
        //     const int index = std::min(static_cast<int>(distance_form_the_origin * base_radii.size()), static_cast<int>(base_radii.size() - 1));
        //     const int radius = base_radii[index] * scale_factor;
        //     if (radius == 0) {return;}
        //     std::cout << "Point: <" << key_point[0] << ", " << key_point[1] << ">" << "\tNormed distance from the origin: " << distance_form_the_origin << "\tRadius: " << radius << std::endl;
        //
        //     const int x = std::max(0, key_point[1] - radius);
        //     int const y = std::max(0, key_point[0] - radius);
        //     int const width = std::min(n_cols - x, 2 * radius);
        //     int const height = std::min(n_rows - y, 2 * radius);
        //
        //     const cv::Rect roi(x, y, width, height);
        //
        //     cv::Mat roi_image = blurred_gray_picture(roi);
        //
        //     const int odd_rad = (radius & 1) ? radius : radius + 1;
        //
        //     cv::GaussianBlur(roi_image, roi_image, cv::Size(odd_rad, odd_rad), odd_rad);
        //
        // }
        //
        // static double compute_orientation(point point, cv::Mat& image, const int& n_cols, const int& n_rows) {
        //     std::vector<std::pair<int, int>> predefined_point_for_matching = {
        //         {33, 0}, {17, -30}, {-17, -30}, {-33, 0}, {-17, 30}, {17, 30},
        //         {22, 13}, {22, -13}, {0, -26}, {-22, -13}, {-22, 13}, {0, 26},
        //         {18, 0}, {9, -17}, {-9, -17}, {-18, 0}, {-9, 17}, {9, 17},
        //         {11, 7}, {11, -7}, {0, -13}, {-11, -7}, {-11, 7}, {0, 13},
        //         {8, 0}, {4, -8}, {-4, -8}, {-8, 0}, {-4, 8}, {4, 8},
        //         {5, 3}, {5, -3}, {0, -6}, {-5, -3}, {-5, 3}, {0, 6},
        //         {4, 0}, {2, -4}, {-2, -4}, {-4, 0}, {-2, 4}, {2, 4},
        //         {0, 0}}; // somebody just use less points for rotation measuring: [(0, 2), (1, 3), (2, 4), (3, 5), (0, 4), (1, 5)]
        //     const size_t M = 903;
        //
        //     double O_x = 0;
        //     double O_y = 0;
        //
        //     for (int i = 0; i < predefined_point_for_matching.size(); i++) {
        //         const double point1_intensity = image.at<double>(std::get<1>(predefined_point_for_matching[i]) + point.y, std::get<0>(predefined_point_for_matching[i]) + point.x);
        //         for (int j = i + 1; j < predefined_point_for_matching.size(); j++) {
        //             const double intensity_change = point1_intensity - image.at<double>(std::get<1>(predefined_point_for_matching[j]) + point.y, std::get<0>(predefined_point_for_matching[j]) + point.x);
        //             // norm of 2 vectors
        //             const double norm = sqrt(std::pow(std::get<1>(predefined_point_for_matching[i]) + point.y - std::get<1>(predefined_point_for_matching[j]) + point.y, 2) +
        //                 std::pow(std::get<0>(predefined_point_for_matching[i]) + point.x - std::get<0>(predefined_point_for_matching[j]) + point.x, 2));
        //             O_x += intensity_change * (std::get<0>(predefined_point_for_matching[i]) - std::get<0>(predefined_point_for_matching[j])) / norm;
        //             O_y += intensity_change * (std::get<1>(predefined_point_for_matching[i]) - std::get<1>(predefined_point_for_matching[j])) / norm;
        //         }
        //     }
        //
        //     return std::atan2(O_y, O_x); // div by M?? Nah, I'd win
        //
        // }
        //
        // static void add_transponed_vector(std::vector<std::vector<int>>& array, const std::vector<double>& add_vector, const size_t index, const size_t num_of_keypoints) {
        //
        //     for (size_t i = 0; i < num_of_keypoints; i++) {
        //
        //         array[i][index] = static_cast<int>(add_vector[i]);
        //         // std::cout << "Array: " << array[i][index] << std::endl;
        //
        //     }
        //     std::cout << "Transposed something" << std::endl;
        //
        // }
        //
        //
        // static auto FREAK_feature_description(const std::vector<std::tuple<int, int, double>>& key_points, cv::Mat blurred_gray_picture, const int& n_cols, const int& n_rows, const double corr_threshold) {
        //
        //     const size_t num_of_keypoints = key_points.size();
        //     const size_t DESCRIPTOR_SIZE = 512;
        //
        //     std::vector<std::pair<int, int>> predefined_point_for_matching = {
        //         {33, 0}, {17, -30}, {-17, -30}, {-33, 0}, {-17, 30}, {17, 30},
        //         {22, 13}, {22, -13}, {0, -26}, {-22, -13}, {-22, 13}, {0, 26},
        //         {18, 0}, {9, -17}, {-9, -17}, {-18, 0}, {-9, 17}, {9, 17},
        //         {11, 7}, {11, -7}, {0, -13}, {-11, -7}, {-11, 7}, {0, 13},
        //         {8, 0}, {4, -8}, {-4, -8}, {-8, 0}, {-4, 8}, {4, 8},
        //         {5, 3}, {5, -3}, {0, -6}, {-5, -3}, {-5, 3}, {0, 6},
        //         {4, 0}, {2, -4}, {-2, -4}, {-4, 0}, {-2, 4}, {2, 4},
        //         {0, 0}};
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
        //     add_transponed_vector(output_matrix, matrix_M[0], 0, num_of_keypoints + 1);
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
        //         add_transponed_vector(output_matrix, matrix_M[min_index], column_count, num_of_keypoints);
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

        //////////////////////////////////////////////////////////////////////////////////////FREAKS

// };




auto test_opencv_sobel(const std::string& filename, const std::string& cur_path) {

    cv::Mat image = cv::imread(cur_path + filename);

    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size(7, 7), 0);
    cv::Mat my_blurred_gray = CornerDetection::custom_bgr2gray(blurred);
    cv::Mat my_gray_regular = CornerDetection::custom_bgr2gray(image);

    cv::Mat Jx, Jy, Jxy, sobelFiltered;

    cv::Sobel(my_blurred_gray, Jx, CV_64F, 1, 0, 3);  // First-order derivative in x
    cv::Sobel(my_blurred_gray, Jy, CV_64F, 0, 1, 3);  // First-order derivative in y
    cv::Sobel(Jx, Jxy, CV_64F, 0, 1, 3);  // Second-order derivative Jxy

    // Compute Sobel magnitude for visualization
    cv::Mat sobelMagnitude;
    cv::magnitude(Jx, Jy, sobelMagnitude);
    sobelMagnitude.convertTo(sobelFiltered, CV_8U);

    // Convert derivatives to displayable format
    cv::Mat Jx_disp, Jy_disp, Jxy_disp;
    cv::convertScaleAbs(Jx, Jx_disp);
    cv::convertScaleAbs(Jy, Jy_disp);
    cv::convertScaleAbs(Jxy, Jxy_disp);

    // Show results
    // cv::imshow("Original", picture);
    cv::imshow("Jx (Gradient X)", Jx_disp);
    cv::imshow("Jy (Gradient Y)", Jy_disp);
    cv::imshow("Jxy (Second Order XY)", Jxy_disp);
    cv::imshow("Sobel Filtered", sobelFiltered);

    return sobelFiltered;

}

auto test_opencv_corner_detection(const std::string& filename, const std::string& cur_path, const std::string& win_name, const int& N) {
    cv::Mat image = cv::imread(cur_path + filename);
    cv::Mat gray;
    std::vector<cv::Point2f> corners;
    if (image.channels() > 1) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    cv::goodFeaturesToTrack(gray, corners, N, 0.01, 20);

    for (const auto& corner : corners) {
        cv::circle(image, corner, 3, cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow(win_name, image);
    cv::imwrite("../test_images/output_images/" + win_name + ".png", image);

}


    // std::cout << "Length: " << v.size() << std::endl;
    // std::string cur_path = __FILE__;
    // cur_path = cur_path.erase(cur_path.length() - 22); //weird staff
    // std::string test_file = "test_images/Zhovkva2.jpg";
    // // std::cout << "CUr path " << cur_path + "test_images/Notre-Dame-de-Paris-France.webp" <<std::endl;
    //
    // // extract_features("test_images/Notre-Dame-de-Paris-France.webp", cur_path);
    // // test_their_sobel("test_images/Notre-Dame-de-Paris-France.webp", cur_path);
    // // test_opencv_sobel("test_images/Notre-Dame-de-Paris-France.webp", cur_path);
    //
    // // CornerDetection::compare_images(CornerDetection::direction_gradients(test_file, cur_path), test_opencv_sobel(test_file, cur_path), "6 mul and opencv");
    // //
    // // CornerDetection::compare_images(CornerDetection::direction_gradients(test_file, cur_path), CornerDetection::test_their_sobel(test_file, cur_path), "9 mul and 6 mul");
    // //
    // // CornerDetection::compare_images(CornerDetection::test_their_sobel(test_file, cur_path), test_opencv_sobel(test_file, cur_path), "opencv and 9 mul");
    //
    // // auto gradients = CornerDetection::direction_gradients(test_file, cur_path);
    // //
    // // int n_rows = gradients[0].rows;
    // // int n_cols = gradients[0].cols;
    // //
    // // cv::imshow("Bohdan sobel rebuilt", CornerDetection::sobel_filter(gradients[0], gradients[1], n_rows, n_cols));
    //
    // CornerDetection::prepare_and_test(test_file, cur_path, "No smoothing", true);
    //
    // // test_opencv_corner_detection(test_file, cur_path, "opencv implementation", 1500);
    //
    //
    // cv::waitKey(0);
    //
    // cv::destroyAllWindows();
    //
    // const std::vector<int> v =
    // {
    //     404,431,818,511,181,52,311,874,774,543,719,230,417,205,11,
    //     560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
    //     592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
    //     796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
    //     691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
    //     381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
    //     382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
    //     466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
    //     418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
    //     72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
    //     56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
    //     129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
    //     236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
    //     769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
    //     544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
    //     212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
    //     194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
    //     276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
    //     844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
    //     736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
    //     182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
    //     242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
    //     819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
    //     185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
    //     851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
    //     13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
    //     413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
    //     197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
    //     41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
    //     152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
    //     260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
    //     131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
    //     325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
    //     670,249,36,581,389,605,331,518,442,822
    // };

