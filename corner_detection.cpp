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
#include <queue>

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

            if (R_array[i][j] <= threshold) { // put the bound for patch checking here
                    // std::cout << "R > " << threshold << ": (" << i << ", " << j << ")" << std::endl;
                    R_array[i][j] = 0;
            }

        }
    }
    return R_array;

}


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
    // std::cout << "COunt: " << count << std::endl; // debug
    return output_corners;
}


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
