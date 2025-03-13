//
// Created by julfy1 on 2/1/25.
//
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <bit>

struct point {
    size_t x, y;
};

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



        static auto test_their_sobel(const std::string& filename, const std::string& cur_path) { // Not needed, but there IDK
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

        static auto direction_gradients(cv::Mat& picture, const int& n_rows, const int& n_cols) {

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

        static auto shitomasi_corner_detection(cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy, const int& n_rows, const int& n_cols, const double& k) {


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


        static auto draw_score_distribution(const std::vector<std::vector<double>>& R_values, const std::string& win_name) {

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


    static auto non_maximum_suppression(std::vector<std::vector<double>> R_values, const int& n_rows, const int& n_cols, const int& k, const int& N) {
            std::priority_queue<std::tuple<double, int, int>> max_heap; // Store (R_value, i, j)
            std::vector<std::tuple<int, int, double>> output_corners;
            output_corners.reserve(N);
            int count = 0;

            for (int i = k / 2; i < n_rows - k / 2; i++) {
                for (int j = k / 2; j < n_cols - k / 2; j++) {
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
                output_corners.push_back({std::get<1>(max_heap.top()), std::get<2>(max_heap.top()), std::get<0>(max_heap.top())});
                max_heap.pop();
                count++;
            }
            std::cout << "COunt: " << count << std::endl;
            return output_corners;
        }
        const double largets_radius = 2 / 3;

        const double smallest_radius = 2 / 24;

        const double retinal_spacing = (largets_radius - smallest_radius) / 21;
    
        const std::vector<double> retinal_keypoint_radii = {largets_radius, largets_radius - 6 * smallest_radius, largets_radius - 11 * smallest_radius,
                         largets_radius - 15 * smallest_radius, largets_radius - 18 * smallest_radius,
                         largets_radius - 20 * smallest_radius, smallest_radius, 0};

        const std::vector<double> retinal_distances_from_the_center = {largets_radius / 2.0, (largets_radius - 6 * smallest_radius) / 2.0, (largets_radius - 11 * smallest_radius) / 2.0,
                     (largets_radius - 15 * smallest_radius) / 2.0, (largets_radius - 18 * smallest_radius) / 2.0,
                     (largets_radius - 20 * smallest_radius) / 2.0, smallest_radius / 2.0, 0};
        //////////////////////////////////////////////////////////////////////////////////////FREAKS

        inline static auto prepare_the_surroundings(const cv::Mat& blurred_gray_picture, const std::vector<int>& key_point, const int& n_cols, const int& n_rows) {
            const std::vector<int> base_radii = {0, 2, 3, 5, 7, 10, 15}; // might be 1 to always blur the surroundings

            const double scale_factor = std::min(n_rows, n_cols) / 100; //22.5

            const double x_norm = static_cast<double>(key_point[1]) / static_cast<double>(n_cols);
            const double y_norm = static_cast<double>(key_point[0]) / static_cast<double>(n_rows);


            const double distance_form_the_origin = std::sqrt(std::pow(x_norm - 0.5, 2) + std::pow(y_norm - 0.5, 2));

            const int index = std::min(static_cast<int>(distance_form_the_origin * base_radii.size()), static_cast<int>(base_radii.size() - 1));
            const int radius = base_radii[index] * scale_factor;
            if (radius == 0) {return;}
            std::cout << "Point: <" << key_point[0] << ", " << key_point[1] << ">" << "\tNormed distance from the origin: " << distance_form_the_origin << "\tRadius: " << radius << std::endl;

            const int x = std::max(0, key_point[1] - radius);
            int const y = std::max(0, key_point[0] - radius);
            int const width = std::min(n_cols - x, 2 * radius);
            int const height = std::min(n_rows - y, 2 * radius);

            const cv::Rect roi(x, y, width, height);

            cv::Mat roi_image = blurred_gray_picture(roi);

            const int odd_rad = (radius & 1) ? radius : radius + 1;

            cv::GaussianBlur(roi_image, roi_image, cv::Size(odd_rad, odd_rad), odd_rad);

        }

        static double compute_orientation(point point, cv::Mat& image, const int& n_cols, const int& n_rows) {
            std::vector<std::pair<int, int>> predefined_point_for_matching = {
                {33, 0}, {17, -30}, {-17, -30}, {-33, 0}, {-17, 30}, {17, 30},
                {22, 13}, {22, -13}, {0, -26}, {-22, -13}, {-22, 13}, {0, 26},
                {18, 0}, {9, -17}, {-9, -17}, {-18, 0}, {-9, 17}, {9, 17},
                {11, 7}, {11, -7}, {0, -13}, {-11, -7}, {-11, 7}, {0, 13},
                {8, 0}, {4, -8}, {-4, -8}, {-8, 0}, {-4, 8}, {4, 8},
                {5, 3}, {5, -3}, {0, -6}, {-5, -3}, {-5, 3}, {0, 6},
                {4, 0}, {2, -4}, {-2, -4}, {-4, 0}, {-2, 4}, {2, 4},
                {0, 0}}; // somebody just use less points for rotation measuring: [(0, 2), (1, 3), (2, 4), (3, 5), (0, 4), (1, 5)]
            const size_t M = 903;

            double O_x = 0;
            double O_y = 0;

            for (int i = 0; i < predefined_point_for_matching.size(); i++) {
                const double point1_intensity = image.at<double>(std::get<1>(predefined_point_for_matching[i]) + point.y, std::get<0>(predefined_point_for_matching[i]) + point.x);
                for (int j = i + 1; j < predefined_point_for_matching.size(); j++) {
                    const double intensity_change = point1_intensity - image.at<double>(std::get<1>(predefined_point_for_matching[j]) + point.y, std::get<0>(predefined_point_for_matching[j]) + point.x);
                    // norm of 2 vectors
                    const double norm = sqrt(std::pow(std::get<1>(predefined_point_for_matching[i]) + point.y - std::get<1>(predefined_point_for_matching[j]) + point.y, 2) +
                        std::pow(std::get<0>(predefined_point_for_matching[i]) + point.x - std::get<0>(predefined_point_for_matching[j]) + point.x, 2));
                    O_x += intensity_change * (std::get<0>(predefined_point_for_matching[i]) - std::get<0>(predefined_point_for_matching[j])) / norm;
                    O_y += intensity_change * (std::get<1>(predefined_point_for_matching[i]) - std::get<1>(predefined_point_for_matching[j])) / norm;
                }
            }

            return std::atan2(O_y, O_x); // div by M?? Nah, I'd win

        }

        static auto FREAK_feature_description(const std::vector<std::tuple<int, int, double>>& key_points, cv::Mat blurred_gray_picture, const int& n_cols, const int& n_rows) {

            for (auto key_point : key_points) {
                prepare_the_surroundings(blurred_gray_picture, {std::get<0>(key_point), std::get<1>(key_point)}, n_cols, n_rows);
            }

            std::vector<std::pair<int, int>> predefined_point_for_matching = {
                {33, 0}, {17, -30}, {-17, -30}, {-33, 0}, {-17, 30}, {17, 30},
                {22, 13}, {22, -13}, {0, -26}, {-22, -13}, {-22, 13}, {0, 26},
                {18, 0}, {9, -17}, {-9, -17}, {-18, 0}, {-9, 17}, {9, 17},
                {11, 7}, {11, -7}, {0, -13}, {-11, -7}, {-11, 7}, {0, 13},
                {8, 0}, {4, -8}, {-4, -8}, {-8, 0}, {-4, 8}, {4, 8},
                {5, 3}, {5, -3}, {0, -6}, {-5, -3}, {-5, 3}, {0, 6},
                {4, 0}, {2, -4}, {-2, -4}, {-4, 0}, {-2, 4}, {2, 4},
                {0, 0}};

            std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> all_point_combinations;

            for (int i = 0; i < predefined_point_for_matching.size(); i++) {

                for (int j = 1; j < predefined_point_for_matching.size(); j++) {
                    all_point_combinations.emplace_back(predefined_point_for_matching[i], predefined_point_for_matching[i]);
                }
            }


            std::cout << "Num of cols: " << n_cols << std::endl;
            std::cout << "Num of rows: " << n_rows << std::endl;
            for (auto key_point : key_points) {
                const auto p = point(std::get<1>(key_point), std::get<0>(key_point));
                if (!((p.x >= 34) && (p.x <= n_cols - 34) && (p.y >= 30) && (p.y <= n_rows - 30))) {
                    // std::cout << "Point to skip: " << "(" << p.x << ", " << p.y << ")" << std::endl ;
                    continue;
                }

                // Rotate the image in the direction of the stronger intensity(for rotation invariance)

                double angle = compute_orientation(p, blurred_gray_picture, n_cols, n_rows);
                // std::cout << "ANgle: " << angle << std::endl;
                cv::putText(blurred_gray_picture, std::to_string(angle), cv::Point(std::get<1>(key_point), std::get<0>(key_point)), cv::FONT_ITALIC, 0.5, cv::Scalar(0, 255, 0), 3);

                const double rotation_matrix[4] = {std::cos(angle), -1 * std::sin(angle), std::sin(angle), std::cos(angle)};

                // Create meaningful pair descriptions

                for (auto pt : predefined_point_for_matching) {
                    point pnt = point(std::get<0>(pt) * rotation_matrix[0] + std::get<1>(pt) * rotation_matrix[2],
                                        (-1) * std::get<0>(pt) * rotation_matrix[1] + std::get<1>(pt) * rotation_matrix[4]);



                }






                // bool point_surroundings_comparison[43];

                     //        sort(vec.begin(), vec.end(),
                     // [](const Complex& a, const Complex& b) {
                     //     // Compare Complex numbers based on their real
                     //     // parts
                     //     if (a.real == b.real) {
                     //         // If real parts are equal, compare based
                     //         // on imaginary parts
                     //         return a.imag < b.imag;
                     //     }
                     //     // If real parts are different, use them for
                     //     // comparison
                     //     return a.real < b.real;
                     // });
            }

            // for ()
            cv::imshow("BOhdan with description", blurred_gray_picture);

        }

        //////////////////////////////////////////////////////////////////////////////////////FREAKS

        static void prepare_and_test(const std::string& filename, const std::string& cur_path, const std::string& win_name, const bool draw = false) {

            cv::Mat image1 = cv::imread(cur_path + filename);

            cv::Mat image2 = cv::imread(cur_path + filename);


            cv::Mat my_blurred_gray;

            const cv::Mat blurred = FeatureExtraction::custom_bgr2gray(image1);

            cv::GaussianBlur(blurred, my_blurred_gray, cv::Size(7, 7), 0);

            int n_rows = my_blurred_gray.rows;
            int n_cols = my_blurred_gray.cols;

            auto gradients = FeatureExtraction::direction_gradients(my_blurred_gray, n_rows, n_cols);

            auto harris_corners = FeatureExtraction::harris_corner_detection(gradients[0], gradients[1], gradients[2], n_rows, n_cols, 0.05);

            auto shitomasi_corners = FeatureExtraction::shitomasi_corner_detection(gradients[0], gradients[1], gradients[2], n_rows, n_cols, 0.05);

            auto local_mins_shitomasi = FeatureExtraction::non_maximum_suppression(shitomasi_corners, n_rows, n_cols, 20, 1500);

            auto local_mins_harris = FeatureExtraction::non_maximum_suppression(harris_corners, n_rows, n_cols, 20, 1500);

            draw_score_distribution(harris_corners, "harris");

            draw_score_distribution(shitomasi_corners, "shi-tomasi");

            FREAK_feature_description(local_mins_shitomasi, blurred, n_cols, n_rows);

            if (draw) {
                for (auto coords : local_mins_shitomasi) {

                    // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

                    cv::circle(image1, cv::Point(std::get<1>(coords), std::get<0>(coords)), 1, cv::Scalar(0, 0, 255), 3);
                }
                cv::imshow("BOhdan with corners shi-tomasi", image1);
                cv::imwrite("../test_images/output_images/shi-tomasi_with_corners.png", image1);
            }

            if (draw) {
                for (auto coords : local_mins_harris) {
                    // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

                    cv::circle(image2, cv::Point(std::get<1>(coords), std::get<0>(coords)), 1, cv::Scalar(0, 0, 255), 3);
                }
                cv::imshow("BOhdan with corners harris", image2);
                cv::imwrite("../test_images/output_images/harris_with_corners.png", image2);
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


int main() {

    std::string cur_path = __FILE__;
    cur_path = cur_path.erase(cur_path.length() - 22); //weird staff
    std::string test_file = "test_images/Zhovkva2.jpg";
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

    FeatureExtraction::prepare_and_test(test_file, cur_path, "No smoothing", true);

    // test_opencv_corner_detection(test_file, cur_path, "opencv implementation", 1500);


    cv::waitKey(0);

    cv::destroyAllWindows();
}