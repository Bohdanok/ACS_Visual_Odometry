//
// Created by admin on 13.04.2025.
//
#pragma once
// #define PRINT_INTERMEDIATE_STEPS

#ifndef POSEUPDATE_H
#define POSEUPDATE_H


#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

// using namespace cv;
extern cv::Mat flipZ;


cv::Vec3d rotationMatrixToEulerAngles(const cv::Mat &R);

cv::Mat readGTLine(const std::string &line);
void writePoseCSV(const std::string &filename, const std::vector<cv::Mat> &poses);

class PoseUpdate
{
private:
    cv::Mat W = (cv::Mat_<double>(3, 3) <<
    0, -1, 0,
    1,  0, 0,
    0,  0, 1);

    cv::Mat K = (cv::Mat_<double>(3,3) <<
    7.188560000000e+02, 0, 6.071928000000e+02,
    0, 7.188560000000e+02, 1.852157000000e+02,
    0, 0, 1.000000000000e+00); // Intrinsic matrix

    // cv::Mat T_BS = (cv::Mat_<double>(4, 4) <<
    //     0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
    //      0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
    //     -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
    //      0.0, 0.0, 0.0, 1.0);

    // cv::Mat T_L = (cv::Mat_<double>(4, 4) <<
    //     1.0, 0.0, 0.0,  7.48903e-02,
    //      0.0, 1.0, 0.0, -1.84772e-02,
    //      0.0, 0.0, 1.0, -1.20209e-01,
    //      0.0, 0.0, 0.0,  1.0);

    // cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

    // cv::Mat R_CS = (cv::Mat_<double>(3, 3) <<
    // 0, 0, 1,
    // 0,  -1, 0,
    // 1, 0, 0);

public:
    std::pair<cv::Mat, cv::Mat> getPose(cv::Mat F, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2,
                                                double GT_pos_norm) {
    // Ensure F is 64-bit double
    K.convertTo(K, CV_64F);
    F.convertTo(F, CV_64F);

    cv::Mat E = K.t() * F * K;

    E /= cv::norm(E);

    if (cv::countNonZero(E) < 5) {
        throw std::runtime_error("Degenerate essential matrix");
    }

    cv::Mat U, S, Vt;
    cv::SVD::compute(E, S, U, Vt);
#ifdef PRINT_INTERMEDIATE_STEPS
    std::cout << "Singular values of E: " << S.t() << std::endl;
#endif

    if (cv::determinant(U) < 0) U *= -1;
    if (cv::determinant(Vt) < 0) Vt *= -1;

    // Compute possible rotation matrices (R1 and R2)
    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;
    if (cv::determinant(R1) < 0) R1 = -R1;
    if (cv::determinant(R2) < 0) R2 = -R2;

    // Get the translation vector from the essential matrix
    cv::Mat t = U.col(2);

    // Store all possible combinations of R and t
    std::vector<std::pair<cv::Mat, cv::Mat>> candidates = {
        {R1, t}, {R1, -t}, {R2, t}, {R2, -t}
    };

    int maxPositiveDepth = -1;
    cv::Mat R_final, t_final;

    for (const auto& [R, t_candidate] : candidates) {
        // Project points using the current rotation and translation candidates
        cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
        cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
        R.copyTo(P2.colRange(0, 3));
        t_candidate.copyTo(P2.col(3));

        // Undistort points
        std::vector<cv::Point2f> normPoints1, normPoints2;
        cv::undistortPoints(points1, normPoints1, K, cv::noArray());
        cv::undistortPoints(points2, normPoints2, K, cv::noArray());

        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, normPoints1, normPoints2, points4D);

        // cv::triangulatePoints(P1, P2,
        //     cv::Mat(normPoints1).t(),
        //     cv::Mat(normPoints2).t(),
        //     points4D);

        // Convert to 64F for precision
        cv::Mat points4D_64F;
        points4D.convertTo(points4D_64F, CV_64F);

        int countPositiveDepth = 0;
        for (int i = 0; i < points4D_64F.cols; i++) {
            cv::Mat x = points4D_64F.col(i);
            double w = x.at<double>(3);
            if (std::abs(w) < 1e-6) continue;
            x /= w;  // Homogenize the coordinates

            double z1 = x.at<double>(2);
            cv::Mat X = x.rowRange(0, 3);
            cv::Mat X2 = R * X + t_candidate;
            double z2 = X2.at<double>(2);

            if (z1 > 0 && z2 > 0) countPositiveDepth++;
        }
#ifdef PRINT_INTERMEDIATE_STEPS
        std::cout << "Candidate R, t:  " << countPositiveDepth << " inliers\n";
#endif
        if (countPositiveDepth > maxPositiveDepth) {
            maxPositiveDepth = countPositiveDepth;
            R_final = R.clone();
            t_final = t_candidate.clone();
        }
    }

        if (maxPositiveDepth < 100)
        {
            std::cerr << "Max positive depth too small\n";
        }

        // cv::Mat T_world = cv::Mat::eye(4, 4, CV_64F);

        if (cv::determinant(R_final) < 0)
            R_final = -R_final;
#ifdef PRINT_INTERMEDIATE_STEPS
            std::cout << "\nChanged the sign\n" << std::endl;
#endif

        // R_final.copyTo(T_world(cv::Rect(0, 0, 3, 3)));

        double t_body_norm = cv::norm(t_final);
        if (t_body_norm > 1e-6) {
            t_final *= (GT_pos_norm / t_body_norm);
        } else {
            std::cerr << "Warning: Translation norm too small, skipping scaling." << std::endl;
        }
        // t_final.copyTo(T_world(cv::Rect(3, 0, 1, 3)));


        // Normalize rotation?
        cv::Mat U1, W, Vt1;
        cv::SVDecomp(R_final, W, U1, Vt1);
        cv::Mat R_world_ = U1 * Vt1;

        return {R_final, t_final};
}

};

cv::Mat quat_to_rot(double w, double x, double y, double z);

#endif //POSEUPDATE_H