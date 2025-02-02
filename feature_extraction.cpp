//
// Created by julfy1 on 2/1/25.
//
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>

auto custom_bgr2gray(cv::Mat& picture) {
    const int n_rows = picture.rows;
    const int n_cols = picture.cols * 3;
    uchar* ptr_src;
    uchar* ptr_dst;
    cv::Mat output_picture = cv::Mat::zeros(n_rows, n_cols / 3, CV_8UC1);

    for(size_t i = 0; i < n_rows; i++) {
        ptr_src = picture.ptr<uchar>(i);
        ptr_dst = output_picture.ptr<uchar>(i);
        for(size_t j = 0; j < n_cols; j += 3) {
            ptr_dst[j / 3] = cv::saturate_cast<uchar>(0.114 * ptr_src[j] + 0.587 * ptr_src[j + 1] + 0.299 * ptr_src[j + 2]);
        }
    }
    return output_picture;

}

auto extract_features(const std::string& filename, const std::string& cur_path) {

    cv::Mat image = cv::imread(cur_path + filename);

    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 0);
    cv::Mat my_blurred_gray = custom_bgr2gray(blurred);
    cv::Mat my_gray_regular = custom_bgr2gray(image);

    cv::imshow("BOHDAN GRAY", my_gray_regular);
    cv::imshow("BOHDAN BLURRED GRAY", my_blurred_gray);

    // for(size_t y = 0; y < gray_blurred.rows; y++) {
    //
    //     uchar* rowPtr = gray_blurred.ptr<uchar>(y);
    //
    //     for(size_t x = 0; x < gray_blurred.cols; x++) {
    //
    //         std::cout << (int)rowPtr[x] << " ";
    //
    //     }
    //     std::cout << "\n";
    // }


    // cv::imshow("BOHDAN REGULAR", image);
    // cv::imshow("BOHDAN GRAY BLURRED", gray_blurred);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}

int main() {

    std::string cur_path = __FILE__;
    cur_path = cur_path.erase(cur_path.length() - 22); //weird staff

    // std::cout << "CUr path " << cur_path + "test_images/Notre-Dame-de-Paris-France.webp" <<std::endl;

    extract_features("test_images/Notre-Dame-de-Paris-France.webp", cur_path);
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
}