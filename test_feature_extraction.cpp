//
// Created by julfy1 on 3/24/25.
//

#include "test_feature_extraction.h"
#include "corner_detection.h"
#include "FREAK_feature_descriptor.h"

#include <iostream>

// #define DRAW_A_VISUALIZATION

void compare_images(const cv::Mat& image_my, const cv::Mat& image_their, const std::string& win_name) {

    cv::Mat output_check, return_something;

    // cv::bitwise_xor(image_my, image_their, output_check);
    cv::subtract(image_their, image_my, return_something);

    cv::imshow(win_name, return_something);
}

void draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name) {

    int rows = R_values.size();
    int cols = R_values[0].size();

    cv::Mat mat(rows, cols, CV_64F); // Create matrix to store values

    // Copy values
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat.at<double>(i, j) = R_values[i][j];
        }
    }

    // Normalize values to range [0, 1]
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Mat normMat = (mat - minVal) / (maxVal - minVal); // Normalize between 0-1

    // Create color image
    cv::Mat colorImage(rows, cols, CV_8UC3);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = normMat.at<double>(i, j); // Normalized value [0, 1]

            // Map to RGB colors (Green → Blue → Red)
            uchar red   = static_cast<uchar>(255 * std::max(0.0, (val - 0.5) * 2));  // Increase red for higher values
            uchar blue  = static_cast<uchar>(255 * std::max(0.0, (0.5 - std::abs(val - 0.5)) * 2));  // Max in the middle
            uchar green = static_cast<uchar>(255 * std::max(0.0, (0.5 - val) * 2));  // Decrease green as value increases

            colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        }
    }
    cv::imshow(win_name, colorImage);
    cv::imwrite("../test_images/output_images/" + win_name + ".png", colorImage);

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

int main() {

    std::string cur_path = __FILE__;
    cur_path = cur_path.erase(cur_path.length() - 27); //weird staff
    std::string test_file = "test_images/Zhovkva2.jpg";

    cv::Mat image1 = cv::imread(cur_path + test_file);

    cv::Mat image2 = cv::imread(cur_path + test_file);


    cv::Mat my_blurred_gray;

    const cv::Mat blurred = CornerDetection::custom_bgr2gray(image1);

    cv::GaussianBlur(blurred, my_blurred_gray, cv::Size(7, 7), 0);
    std::cout << "Hi";
    int n_rows = my_blurred_gray.rows;
    int n_cols = my_blurred_gray.cols;

    auto gradients = CornerDetection::direction_gradients(my_blurred_gray, n_rows, n_cols);

    auto harris_corners = CornerDetection::harris_corner_detection(gradients[0], gradients[1], gradients[2], n_rows, n_cols, 0.05);

    auto shitomasi_corners = CornerDetection::shitomasi_corner_detection(gradients[0], gradients[1], gradients[2], n_rows, n_cols, 0.05);

    auto local_mins_shitomasi = CornerDetection::non_maximum_suppression(shitomasi_corners, n_rows, n_cols, 20, 1500);

    auto local_mins_harris = CornerDetection::non_maximum_suppression(harris_corners, n_rows, n_cols, 20, 1500);

    draw_score_distribution(harris_corners, "harris");

    draw_score_distribution(shitomasi_corners, "shi-tomasi");

    // FREAK::FREAK_feature_description(local_mins_shitomasi, blurred, n_cols, n_rows, 0.5);
    #ifdef DRAW_A_VISUALIZATION

        for (auto coords : local_mins_shitomasi) {

            // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

            cv::circle(image1, cv::Point(coords.x, coords.y), 1, cv::Scalar(0, 0, 255), 3);
        }
        cv::imshow("BOhdan with corners shi-tomasi", image1);
        cv::imwrite("../test_images/output_images/shi-tomasi_with_corners.png", image1);


        for (auto coords : local_mins_harris) {
            // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

            cv::circle(image2, cv::Point(coords.x, coords.y), 1, cv::Scalar(0, 0, 255), 3);
        }
        cv::imshow("BOhdan with corners harris", image2);
        cv::imwrite("../test_images/output_images/harris_with_corners.png", image2);

    cv::waitKey(0);

    cv::destroyAllWindows();

#endif

    auto descriptor = FREAK::FREAK_feature_description(local_mins_shitomasi, blurred, n_cols, n_rows);
    std::cout << "Hi from test feature extraction before!" << std::endl;

    print_descriptor(descriptor);

}