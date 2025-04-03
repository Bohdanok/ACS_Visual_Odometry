//
// Created by julfy1 on 3/24/25.
//

#include "test_feature_extraction.h"
#include "corner_detection.h"
#include "FREAK_feature_descriptor.h"

#include <iostream>

#define DRAW_A_VISUALIZATION

void draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name) {

    int rows = R_values.size();
    int cols = R_values[0].size();

    cv::Mat mat(rows, cols, CV_64F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat.at<double>(i, j) = R_values[i][j];
        }
    }

    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Mat normMat = (mat - minVal) / (maxVal - minVal); // Normalize between 0-1

    cv::Mat colorImage(rows, cols, CV_8UC3);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = normMat.at<double>(i, j); // Normalized value [0, 1]

            uchar red   = static_cast<uchar>(255 * std::max(0.0, (val - 0.5) * 2));  // Increase red for higher values
            uchar blue  = static_cast<uchar>(255 * std::max(0.0, (0.5 - std::abs(val - 0.5)) * 2));  // Max in the middle
            uchar green = static_cast<uchar>(255 * std::max(0.0, (0.5 - val) * 2));  // Decrease green as value increases

            colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        }
    }
    cv::imshow(win_name, colorImage);
    // cv::imwrite("../test_images/output_images/" + win_name + ".png", colorImage);

}

void compare_images(const cv::Mat& image_my, const cv::Mat& image_their, const std::string& win_name) {

    cv::Mat output_check, return_something;

    // cv::bitwise_xor(image_my, image_their, output_check);
    cv::subtract(image_their, image_my, return_something);

    cv::imshow(win_name, return_something);
}

void print_descriptor(const std::vector<std::vector<uint8_t>>& descriptor){
    std::cout << "/////////////////////////////////////////////////////////////" << std::endl;
    for (size_t i = 0; i < descriptor.size(); i++) {
        std::cout << "<";
        for (size_t j = 0; j < descriptor[0].size(); j++) {
            std::cout << static_cast<int>(descriptor[i][j]) << ", ";
        }
        std::cout << ">" << std::endl;
    }
    std::cout << "/////////////////////////////////////////////////////////////" << std::endl;

}

std::vector<std::vector<uint8_t>> descriptor_for_s_pop(const std::string& filename) {

    cv::Mat image1 = cv::imread(filename);

    cv::Mat image2 = cv::imread(filename);

    cv::Mat my_blurred_gray;

    const cv::Mat blurred = CornerDetection::custom_bgr2gray(image1);

    cv::GaussianBlur(blurred, my_blurred_gray, cv::Size(7, 7), 0);

    int n_rows = my_blurred_gray.rows;
    int n_cols = my_blurred_gray.cols;

    auto gradients = CornerDetection::direction_gradients(my_blurred_gray, n_rows, n_cols);

    auto shitomasi_response = CornerDetection::shitomasi_corner_detection(gradients[0], gradients[1], gradients[2], n_rows, n_cols, 0.05);

    auto local_mins_shitomasi = CornerDetection::non_maximum_suppression(shitomasi_response, n_rows, n_cols, 5, 1500);

    // auto local_mins_shitomasi_bohdan = CornerDetection::non_maximum_suppression_bohdan(shitomasi_response, n_rows, n_cols, 5, 1500);


#ifdef DRAW_A_VISUALIZATION
    // draw_score_distribution(shitomasi_response, "Response serial");
    // //
    // // cv::imshow("Jx", gradients[0]);
    // // cv::imshow("Jy", gradients[1]);
    // // cv::imshow("Jxy", gradients[2]);
    // for (auto coords : local_mins_shitomasi_sofia) {
    //
    //     // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;
    //
    //     cv::circle(image2, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
    // }
    //
    // for (auto coords : local_mins_shitomasi_bohdan) {
    //
    //     // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;
    //
    //     cv::circle(image1, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
    // }
    //
    //
    // cv::imshow("NMS Bohdan", image1);
    // cv::imshow("NMS Sofia", image2);
    //
    //
    // cv::waitKey(0);
    // cv::destroyAllWindows();

#endif

    // std::cout << "Number of keypoints: " << local_mins_shitomasi.size() << std::endl;

    auto descriptor = FREAK::FREAK_feature_description(local_mins_shitomasi, blurred, n_cols, n_rows);

    return descriptor;
}

std::vector<std::vector<uint8_t>> descriptor_with_keypoints(const std::string& filename) {

    cv::Mat image1 = cv::imread(filename);

    cv::Mat image2 = cv::imread(filename);

    cv::Mat my_blurred_gray;

    const cv::Mat blurred = CornerDetection::custom_bgr2gray(image1);

    cv::GaussianBlur(blurred, my_blurred_gray, cv::Size(7, 7), 0);

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;

    auto gradients = CornerDetection::direction_gradients(my_blurred_gray, n_rows, n_cols);

    auto shitomasi_corners = CornerDetection::shitomasi_corner_detection(gradients[0], gradients[1], gradients[2], n_rows, n_cols, 0.05);

#ifdef DRAW_A_VISUALIZATION
    draw_score_distribution(shitomasi_corners, "Response serial");

    cv::waitKey(0);
    cv::destroyAllWindows();

#endif


    auto local_mins_shitomasi = CornerDetection::non_maximum_suppression(shitomasi_corners, n_rows, n_cols, 5, 1500);

    std::cout << "Number of keypoints: " << local_mins_shitomasi.size() << std::endl;

    auto descriptor = FREAK::FREAK_feature_description(local_mins_shitomasi, blurred, n_cols, n_rows);

    return descriptor;
}

std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> descriptor_with_points(const std::string& filename) {

    cv::Mat image1 = cv::imread(filename);

    cv::Mat image2 = cv::imread(filename);

    cv::Mat my_blurred_gray;

    const cv::Mat blurred = CornerDetection::custom_bgr2gray(image1);

    cv::GaussianBlur(blurred, my_blurred_gray, cv::Size(7, 7), 0);

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;

    auto gradients = CornerDetection::direction_gradients(my_blurred_gray, n_rows, n_cols);

    auto shitomasi_corners = CornerDetection::shitomasi_corner_detection(gradients[0], gradients[1], gradients[2], n_rows, n_cols, 0.05);
#ifdef DRAW_A_VISUALIZATION
    // draw_score_distribution(shitomasi_corners, "Response serial");
    //
    // cv::waitKey(0);
    // cv::destroyAllWindows();

#endif

    auto local_mins_shitomasi = CornerDetection::non_maximum_suppression(shitomasi_corners, n_rows, n_cols, 5, 1500);

    // std::cout << "Number of key points in serial: " << local_mins_shitomasi.size() << std::endl;

    auto descriptor = FREAK::FREAK_feature_description(local_mins_shitomasi, blurred, n_cols, n_rows);

    return {descriptor, local_mins_shitomasi};

}


 // int main() {
 //
 //     std::string cur_path = __FILE__;
 //     cur_path = cur_path.erase(cur_path.length() - 27); //weird staff
 //     std::string test_file = "test_images/Zhovkva2.jpg";
 //     std::string test_image = "/home/julfy1/Documents/4th_term/ACS/ACS_Visual_Odometry_SOFIA/ACS_Visual_Odometry/images/Zhovkva2.jpg";
 //
 //
 //     auto descriptor = descriptor_for_s_pop(test_image);
 //
 //    // print_descriptor(descriptor);
 //
 // }