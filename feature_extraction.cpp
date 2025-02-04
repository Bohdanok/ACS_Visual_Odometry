//
// Created by julfy1 on 2/1/25.
//
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <cmath>

auto custom_bgr2gray(cv::Mat& picture) {
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

auto test_their_sobel(const std::string& filename, const std::string& cur_path) {
    cv::Mat image = cv::imread(cur_path + filename);

    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size(7, 7), 0);
    cv::Mat my_blurred_gray = custom_bgr2gray(blurred);
    cv::Mat my_gray_regular = custom_bgr2gray(image);

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;

    cv::Mat test_their_sobel = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);

    uchar* ptr_src1;
    uchar* ptr_src2;
    uchar* ptr_src3;
    uchar* ptr_dstx;
    uchar* ptr_dsty;

    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int gy[3][3] = {
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

auto extract_features(const std::string& filename, const std::string& cur_path) {

    cv::Mat image = cv::imread(cur_path + filename);

    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 0);
    cv::Mat my_blurred_gray = custom_bgr2gray(blurred);
    cv::Mat my_gray_regular = custom_bgr2gray(image);

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;
    uchar* ptr_src1;
    uchar* ptr_src2;
    uchar* ptr_src3;
    uchar* ptr_dst_Jx;
    uchar* ptr_dst_Jy;
    int sumx[3] = {0};
    int sumy[3] = {0};
    cv::Mat test_sobel = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);

    cv::Mat Jx = cv::Mat::zeros(n_rows, n_cols, CV_8SC1);
    cv::Mat Jy = cv::Mat::zeros(n_rows, n_cols, CV_8SC1);

    for (size_t i = 1; i < n_rows - 1; i++) {

        ptr_src1 = my_blurred_gray.ptr<uchar>(i - 1);
        ptr_src2 = my_blurred_gray.ptr<uchar>(i);
        ptr_src3 = my_blurred_gray.ptr<uchar>(i + 1);
        ptr_dst_Jx = Jx.ptr<uchar>(i);
        ptr_dst_Jy = Jy.ptr<uchar>(i);

        for (size_t j = 1; j < n_cols - 1; j++) {

            for (short k = -1; k <= 1; k++) {
                sumx[k + 1] = ptr_src1[j + k] - ptr_src3[j + k]; // [1, 0, -1]
                sumy[k + 1] = ptr_src1[j + k] + 2 * ptr_src2[j + k] + ptr_src3[j + k]; // [1, 2, 1]
            }

            ptr_dst_Jx[j] = sumx[0] + 2 * sumx[1] + sumx[2]; // [1, 2, 1] T
            ptr_dst_Jy[j] = sumy[0] - sumy[2]; // [1, 0, -1] T

        }
    }

    double jx2, jy2, jxy, det, trace;
    const double k = 0.5;
    double R;
    double max_R = -DBL_MIN;

    char* ptr_srcjx;
    char* ptr_srcjy;

    for (size_t i = 0; i < n_rows; i++) {

        ptr_srcjx = Jx.ptr<char>(i);
        ptr_srcjy = Jy.ptr<char>(i);

        ptr_src3 = test_sobel.ptr<uchar>(i);

        for (size_t j = 0; j < n_cols; j++) {

            // R=det(M)−k⋅(trace(M))**2
            jx2 = ptr_srcjx[j] * ptr_srcjx[j];
            jy2 = ptr_srcjy[j] * ptr_srcjy[j];
            jxy = ptr_srcjx[j] * ptr_srcjy[j];

            det = (jx2 * jy2) - (jxy * jxy);
            trace = jx2 + jy2;

            R =  det - (k * trace * trace);
            // std::cout << "R: " << R << std::endl;
            max_R = std::max(max_R, R);
            if (R >= 0) {
                std::cout << "R > 0: (" << i << ", " << j << ")" << std::endl;
                cv::circle(image, cv::Point(j, i), 5, cv::Scalar(0, 255, 0), 2);
            }

            ptr_src3[j] = (uchar)(sqrt(ptr_srcjx[j] * ptr_srcjx[j] + ptr_srcjy[j] * ptr_srcjy[j]));

        }

    }


    // cv::imshow("BOHDAN GRAY", my_gray_regular);
    // cv::imshow("BOHDAN BLURRED GRAY", my_blurred_gray);
    std::cout << "Max R: " << max_R << std::endl;
    cv::imshow("BOHDAN JX", Jx);
    cv::imshow("BOHDAN JY", Jy);

    cv::imshow("BOHDAN WITH CORNERS", image);
    cv::imshow("BOHDAN SOBEL", test_sobel);

    // cv::imshow("BOHDAN GRAY SACLE", my_blurred_gray);

    return 0;
}

int main() {

    std::string cur_path = __FILE__;
    cur_path = cur_path.erase(cur_path.length() - 22); //weird staff

    // std::cout << "CUr path " << cur_path + "test_images/Notre-Dame-de-Paris-France.webp" <<std::endl;

    extract_features("test_images/Notre-Dame-de-Paris-France.webp", cur_path);
    test_their_sobel("test_images/Notre-Dame-de-Paris-France.webp", cur_path);
    // std::cout << cv::getBuildInformation() << std::endl;
    // uchar d1 = 'A';
    // uchar d2 = 'A';
    // uchar d3 = d1 + d2;
    //
    // std::cout << (int)d1 << std::endl;
    // std::cout << (int)d2 << std::endl;
    // std::cout << (int)d3 << std::endl;
    //
    // std::cout << (int)(d3 + d3) << std::endl;
    cv::waitKey(0);

    cv::destroyAllWindows();
}