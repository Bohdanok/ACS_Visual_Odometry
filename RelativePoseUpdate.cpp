//
// Created by admin on 16.03.2025.
//
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

class PoseUpdate {
    private:
    cv::Mat W = (cv::Mat_<float>(3, 3) <<
    0, -1, 0,
    1,  0, 0,
    0,  0, 1);

    cv::Mat K = (cv::Mat_<float>(3,3) <<
    458.654, 0, 367.215,
    0, 457.296, 248.375,
    0, 0, 1); // Intrinsics matrix

    public:
    std::pair<cv::Mat, cv::Mat> getPose(cv::Mat F, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, cv::Mat GT_pos1, cv::Mat GT_pos2, Mat R_prev, Mat t_prev) {
        cv::Mat E = K.t() * F * K; // Essential matrix

        cv::Mat U, S, Vt, R_final, t_final;
        cv::SVD::compute(E, S, U, Vt); // decompose E

        if (cv::determinant(U) < 0) U *= -1;
        if (cv::determinant(Vt) < 0) Vt *= -1; // ensure these matrixes are correct rotation matrixes

        cv::Mat R1 = U * W * Vt;
        cv::Mat R2 = U * W.t() * Vt; // compute possible rotations

        if (cv::determinant(R1) < 0) R1 = -R1;
        if (cv::determinant(R2) < 0) R2 = -R2; // Ensure proper rotations

        cv::Mat t = U.col(2); // translation rotation (up to a scale)

        // we get four possible solutions (R1, t), (R1, -t), (R2, t), (R2, -t)
        // triangulate 3D points for each of these 4 combinations and check which
        // one puts points in front of both cameras (images in our case).

        std::vector<std::pair<cv::Mat, cv::Mat>> candidates = {
            {R1,  t},
            {R1, -t},
            {R2,  t},
            {R2, -t}
        };

        int maxPositiveDepth = -1;
        for (const auto& [R, t_candidate] : candidates) {

            // Projection matrices
            cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32F);
            cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
            I.copyTo(P1.colRange(0, 3)); // no rotation and translation in the first image

            cv::Mat P2 = cv::Mat::zeros(3, 4, CV_32F);
            R.copyTo(P2.colRange(0, 3));
            t_candidate.copyTo(P2.col(3)); // rotation and translation to the second image

            std::vector<cv::Point2f> normPoints1, normPoints2;
            cv::undistortPoints(points1, normPoints1, K, cv::Mat());
            cv::undistortPoints(points2, normPoints2, K, cv::Mat());
            // go from the pixel coordinates to normalized (u, v) -> ((u - cx) / fx , (v - cy) / fy)

            // Triangulate points (finding where the two rays
            // (one from each camera through the corresponding pixel) intersect in space)
            cv::Mat points4D;
            cv::triangulatePoints(P1, P2, normPoints1, normPoints2, points4D);
            std::cout << "points4D type: " << points4D.type() << std::endl;

            // count points with positive depth
            // (the biggest amount of points we see from two views identifies the correct R and t)
            int countPositiveDepth = 0;
            for (int i = 0; i < points4D.cols; i++) {
                cv::Mat x = points4D.col(i);
                x /= x.at<float>(3); // Convert from homogeneous coordinates [X, Y, Z, w] to Cartesian coordinates [X/w, Y/w, Z/w].

                float z1 = x.at<float>(2);  // Depth in camera 1

                // Depth in camera 2
                cv::Mat X = x.rowRange(0, 3);
                cv::Mat X2 = R * X + t_candidate;
                float z2 = X2.at<float>(2);

                if (z1 > 0 && z2 > 0) countPositiveDepth++;
            }

            // Check if this is the best so far
            if (countPositiveDepth > maxPositiveDepth) {
                maxPositiveDepth = countPositiveDepth;
                R_final = R.clone();
                t_final = t_candidate.clone();
            }
        }

        cv::Mat GT_transl = GT_pos2 - GT_pos1;
        Mat t_GT = GT_transl.t();
        float scale = norm(t_GT) / norm(t_final); // find a scale

        Mat scaledT = t_final * scale;

        // New pose
        Mat R_new = R_prev * R_final;
        Mat t_new = R_prev * scaledT + t_prev;

        return std::make_pair(R_new, t_new);
    }
};

int main() {
    // Fake example for testing
    cv::Mat F = (cv::Mat_<float>(3, 3) <<
        0, -0.0001, 0.01,
        0.0001, 0, -0.02,
       -0.01, 0.02, 1); // Fake fundamental matrix (just to test numerically)

    // Simulated keypoints (matched pairs)
    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < 50; i++) {
        points1.push_back(cv::Point2f(100 + i, 200 + i)); // Points in image 1
        points2.push_back(cv::Point2f(102 + i, 198 + i)); // Shifted slightly for image 2
    }

    // Ground truth camera positions (3x1)
    cv::Mat GT_pos1 = (cv::Mat_<float>(3, 1) << 0, 0, 0);
    cv::Mat GT_pos2 = (cv::Mat_<float>(3, 1) << 1, 0, 0); // 1 meter apart along X

    // Start with previous pose as identity and zero
    cv::Mat R_prev = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat t_prev = cv::Mat::zeros(3, 1, CV_32F);

    PoseUpdate poseEstimator;
    std::pair<cv::Mat, cv::Mat> result = poseEstimator.getPose(F, points1, points2, GT_pos1, GT_pos2, R_prev, t_prev);
    cv::Mat R_new = result.first;
    cv::Mat t_new = result.second;

    std::cout << "Estimated Rotation:\n" << R_new << std::endl;
    std::cout << "Estimated Translation:\n" << t_new << std::endl;
}
