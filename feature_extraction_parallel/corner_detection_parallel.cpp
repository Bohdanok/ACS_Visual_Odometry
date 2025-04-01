//
// Created by julfy1 on 2/1/25.
//

#include "corner_detection_parallel.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <cmath>
#include <queue>

cv::Mat CornerDetectionParallel::custom_bgr2gray(cv::Mat& picture) {
    const int n_rows = picture.rows;
    const int n_cols = picture.cols * 3;
    cv::Mat output_picture = cv::Mat::zeros(n_rows, n_cols / 3, CV_8UC1);

    for (int i = 0; i < n_rows; i++) {
        const auto *ptr_src = picture.ptr<uchar>(i);
        auto *ptr_dst = output_picture.ptr<uchar>(i);
        for (int j = 0; j < n_cols; j += 3) {
            ptr_dst[j / 3] = cv::saturate_cast<uchar>(0.114 * ptr_src[j] + 0.587 * ptr_src[j + 1] + 0.299 * ptr_src[j + 2]);
        }
    }
    return output_picture;

}

void CornerDetectionParallel::direction_gradients_worker(const cv::Mat& picture, const interval& interval, cv::Mat& Jx, cv::Mat& Jy, cv::Mat& Jxy) {
    double sumx[3] = {0};
    double sumy[3] = {0};

    for (int i = interval.rows.start + 1; i < interval.rows.end - 1; i++) {

        auto *ptr_src1 = picture.ptr<uchar>(i - 1);
        auto *ptr_src2 = picture.ptr<uchar>(i);
        auto *ptr_src3 = picture.ptr<uchar>(i + 1);
        auto *ptr_dst_Jx = Jx.ptr<double>(i);
        auto *ptr_dst_Jy = Jy.ptr<double>(i);
        auto *ptr_dst_Jxy = Jxy.ptr<double>(i);

        for (int j = interval.cols.start + 1; j < interval.cols.end - 1; j++) {

            for (short k = -1; k <= 1; k++) {
                sumx[k + 1] = ptr_src1[j + k] - ptr_src3[j + k]; // [1, 0, -1] T
                sumy[k + 1] = ptr_src1[j + k] + 2 * ptr_src2[j + k] + ptr_src3[j + k]; // [1, 2, 1] T

            }

            ptr_dst_Jx[j] = sumx[0] + 2 * sumx[1] + sumx[2]; // [1, 2, 1]
            ptr_dst_Jy[j] = sumy[0] - sumy[2]; // [1, 0, -1]
            ptr_dst_Jxy[j] = sumx[0] - sumx[2]; // [1, 0, -1]

        }
    }

}


void CornerDetectionParallel::shitomasi_corner_detection_worker(const cv::Mat& Jx, const cv::Mat& Jy, const cv::Mat& Jxy, const interval& interval, const double& k, std::vector<std::vector<double>>& R_array){

    double jx2, jy2;

    for (int i = interval.rows.start + 2; i < interval.rows.end - 2; i++) { // if the length of the interval is 1???!?

        for (int j = interval.cols.start + 2; j < interval.cols.end - 2; j++) {

            // R = det(M)−k⋅(trace(M))**2
            double sumjxy = 0;
            jx2 = 0, jy2 = 0;

            for (int m = -2; m <= 2; m++) {
                const auto *ptr_srcjx = Jx.ptr<double>(i + m);
                const auto *ptr_srcjy = Jy.ptr<double>(i + m);
                const auto *ptr_srcjxy = Jxy.ptr<double>(i + m);

                for (int n = -2; n <= 2; n++) {
                    const double jx = ptr_srcjx[j + n];
                    const double jy = ptr_srcjy[j + n];
                    const double jxy = ptr_srcjxy[j + n];

                    // sumjx += jx;
                    // sumjy += jy;
                    sumjxy += jxy;

                    jx2 += jx * jx;  // Accumulate squared Jx values
                    jy2 += jy * jy;  // Accumulate squared Jy values
                }
            }

            const double det = (jx2 * jy2) - (sumjxy * sumjxy);
            const double trace = jx2 + jy2;

            const double R = (trace / 2) - (0.5 * std::sqrt(trace * trace - 4 * det));

            R_array[i][j] = R > RESPONSE_THRESHOLD ? R : 0;

        }

    }

}




std::vector<cv::KeyPoint> CornerDetectionParallel::non_maximum_suppression(const std::vector<std::vector<double>> &R_values, const int& n_rows, const int& n_cols, const int& k, const int& N) {
    std::priority_queue<std::tuple<double, int, int>> max_heap; // Store (R_value, i, j)
    std::vector<cv::KeyPoint> output_corners;
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
        output_corners.push_back({cv::Point2f(static_cast<float>(std::get<2>(max_heap.top())), static_cast<float>(std::get<1>(max_heap.top()))), 1.0f});
        max_heap.pop();
        count++;
    }
    return output_corners;
}
