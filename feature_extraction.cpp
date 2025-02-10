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

class FeatureExtraction {

    public:
    
        static auto custom_bgr2gray(cv::Mat& picture) {
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



        static auto test_their_sobel(const std::string& filename, const std::string& cur_path) {
            cv::Mat image = cv::imread(cur_path + filename);

            cv::Mat blurred;

            cv::GaussianBlur(image, blurred, cv::Size(7, 7), 0);
            cv::Mat my_blurred_gray = FeatureExtraction::custom_bgr2gray(blurred);
            cv::Mat my_gray_regular = FeatureExtraction::custom_bgr2gray(image);

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

        static void compare_images(const cv::Mat& image_my, const cv::Mat& image_their, const std::string win_name) {

            cv::Mat output_check, return_something;

            // cv::bitwise_xor(image_my, image_their, output_check);
            cv::subtract(image_their, image_my, return_something);

            cv::imshow(win_name, return_something);
        }

        // static auto prepare_image()

        static auto direction_gradients(cv::Mat& picture, const int& n_rows, const int& n_cols) {

            // const int n_rows = my_blurred_gray.rows;
            // const int n_cols = my_blurred_gray.cols;
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

        static auto sobel_filter(cv::Mat& Jx, cv::Mat& Jy, const int& n_rows, const int& n_cols) {
            // const int n_rows = Jx.rows;
            // const int n_cols = Jx.cols;

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

        static auto harris_corner_detection(cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, const int& n_rows, const int& n_cols, const double& k) {

            std::vector<std::vector<int>> feature_array;

            double jx2, jy2, det, trace, R;
            // const double k = 0.07;
            // double max_R = -DBL_MIN;

            double* ptr_srcjx;
            double* ptr_srcjy;
            double* ptr_srcjxy;

            for (int i = 0; i < n_rows; i++) {

                ptr_srcjx = Jx.ptr<double>(i);
                ptr_srcjy = Jy.ptr<double>(i);
                ptr_srcjxy = Jxy.ptr<double>(i);

                for (int j = 0; j < n_cols; j++) {

                    // R = det(M)−k⋅(trace(M))**2
                    jx2 = ptr_srcjx[j] * ptr_srcjx[j];
                    jy2 = ptr_srcjy[j] * ptr_srcjy[j];

                    det = (jx2 * jy2) - (ptr_srcjxy[j] * ptr_srcjxy[j]);
                    trace = jx2 + jy2;

                    R = det - (k * trace * trace);
                    // std::cout << "R: " << R << std::endl;
                    // max_R = std::max(max_R, R);
                    int check_number = 100000;
                    if (R > check_number) {
                        std::cout << "R > " << check_number << ": (" << i << ", " << j << ")" << std::endl;
                        feature_array.push_back({j, i});
                        // cv::circle(original_image, cv::Point(j, i), 0.5, cv::Scalar(0, 255, 0), 1);
                    }

                }

            }


            // cv::imshow("BOHDAN GRAY", my_gray_regular);
            // cv::imshow("BOHDAN BLURRED GRAY", my_blurred_gray);
            // if (show_flag) {
            //
            //     cv::imshow("BOHDAN JX", Jx);
            //     cv::imshow("BOHDAN JY", Jy);
            //     cv::imshow("BOHDAN JXY", Jxy);
            //
            //     // cv::imshow("BOHDAN WITH CORNERS", original_image);
            //
            //
            // }
            return feature_array;

        }

        static void prepare_and_test(const std::string& filename, const std::string& cur_path, const bool draw = false) {

            cv::Mat image = cv::imread(cur_path + filename);

            cv::Mat my_blurred_gray;

            const cv::Mat blurred = FeatureExtraction::custom_bgr2gray(image);

            cv::GaussianBlur(blurred, my_blurred_gray, cv::Size(7, 7), 0);

            int n_rows = my_blurred_gray.rows;
            int n_cols = my_blurred_gray.cols;

            auto gradients = FeatureExtraction::direction_gradients(my_blurred_gray, n_rows, n_cols);

            auto harris_corners = FeatureExtraction::harris_corner_detection(gradients[0], gradients[1], gradients[2], n_rows, n_cols, 0.1);

            if (draw) {
                for (auto coords : harris_corners) {
                    cv::circle(image, cv::Point(coords[0], coords[1]), 0.5, cv::Scalar(0, 255, 0), 1);
                }
                cv::imshow("BOhdan with corners", image);
            }


        }

};


auto test_opencv_sobel(const std::string& filename, const std::string& cur_path) {

    cv::Mat image = cv::imread(cur_path + filename);

    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size(7, 7), 0);
    cv::Mat my_blurred_gray = FeatureExtraction::custom_bgr2gray(blurred);
    cv::Mat my_gray_regular = FeatureExtraction::custom_bgr2gray(image);

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



int main() {

    std::string cur_path = __FILE__;
    cur_path = cur_path.erase(cur_path.length() - 22); //weird staff
    std::string test_file = "test_images/butter.webp";
    // std::cout << "CUr path " << cur_path + "test_images/Notre-Dame-de-Paris-France.webp" <<std::endl;

    // extract_features("test_images/Notre-Dame-de-Paris-France.webp", cur_path);
    // test_their_sobel("test_images/Notre-Dame-de-Paris-France.webp", cur_path);
    // test_opencv_sobel("test_images/Notre-Dame-de-Paris-France.webp", cur_path);

    // FeatureExtraction::compare_images(FeatureExtraction::direction_gradients(test_file, cur_path), test_opencv_sobel(test_file, cur_path), "6 mul and opencv");
    //
    // FeatureExtraction::compare_images(FeatureExtraction::direction_gradients(test_file, cur_path), FeatureExtraction::test_their_sobel(test_file, cur_path), "9 mul and 6 mul");
    //
    // FeatureExtraction::compare_images(FeatureExtraction::test_their_sobel(test_file, cur_path), test_opencv_sobel(test_file, cur_path), "opencv and 9 mul");

    // auto gradients = FeatureExtraction::direction_gradients(test_file, cur_path);
    //
    // int n_rows = gradients[0].rows;
    // int n_cols = gradients[0].cols;
    //
    // cv::imshow("Bohdan sobel rebuilt", FeatureExtraction::sobel_filter(gradients[0], gradients[1], n_rows, n_cols));

    FeatureExtraction::prepare_and_test(test_file, cur_path, true);

    cv::waitKey(0);

    cv::destroyAllWindows();
}