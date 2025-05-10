//
// Created by admin on 13.04.2025.
//

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include "PoseUpdate.hpp"

cv::Mat K = (cv::Mat_<double>(3,3) <<
    7.188560000000e+02, 0, 6.071928000000e+02,
    0, 7.188560000000e+02, 1.852157000000e+02,
    0, 0, 1.000000000000e+00);

void testPoseEstimation(PoseUpdate& poseEstimator)
{
    cv::Mat img1 = cv::imread("../images_examples/000717.png");
    cv::Mat img2 = cv::imread("../images_examples/000718.png");

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return;
    }

    // Feature detection and description
    cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
    freak->compute(img1, keypoints1, descriptors1);
    freak->compute(img2, keypoints2, descriptors2);

    if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "FREAK descriptors empty!" << std::endl;
        return;
    }

    // Matching with ratio test
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    std::vector<cv::DMatch> matches;
    const float ratio_thresh = 0.75f;

    for (const auto& knn_pair : knn_matches) {
        if (knn_pair.size() >= 2 &&
            knn_pair[0].distance < ratio_thresh * knn_pair[1].distance) {
            matches.push_back(knn_pair[0]);
        }
    }

    if (matches.size() < 8) {
        std::cerr << "Not enough good matches!" << std::endl;
        return;
    }

    // Get matched keypoints
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Fundamental matrix
    std::vector<uchar> inliers(points1.size());
    cv::Mat F = findFundamentalMat(points1, points2, cv::FM_RANSAC, 1, 0.99, inliers);
    if (F.empty()) {
        std::cerr << "Fundamental matrix computation failed!" << std::endl;
        return;
    }

    std::vector<cv::Point2f> inlierPoints1, inlierPoints2;
    std::vector<cv::DMatch> inlierMatches;
    for (size_t i = 0; i < points1.size(); ++i) {
        if (inliers[i]) {
            inlierPoints1.push_back(points1[i]);
            inlierPoints2.push_back(points2[i]);
            inlierMatches.push_back(matches[i]);
        }
    }

    std::vector<float> values1 = {8.314209e-01, -2.593021e-02, 5.550378e-01, -2.261366e-01, 3.340032e-02, 9.994364e-01, -3.340532e-03, 2.467171e+00, -5.546384e-01, 2.131582e-02, 8.318184e-01, -1.520606e+02};
    std::vector<float> values2 = {8.661018e-01, -2.579227e-02, 4.992017e-01, 6.230717e-02, 3.056835e-02, 9.995317e-01, -1.392433e-03, 2.456904e+00, -4.989321e-01, 1.646576e-02, 8.664847e-01, -1.514836e+02};
    cv::Mat GT_pos1 = (cv::Mat_<float>(3, 1) <<
        values1[3],
        values1[7],
        values1[11]);
    cv::Mat GT_pos2 = (cv::Mat_<float>(3, 1) <<
        values2[3],
        values2[7],
        values2[11]);

    cv::Mat T_i = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat T_j = cv::Mat::eye(4, 4, CV_64F);

    cv::Mat R_gt1 = (cv::Mat_<float>(3, 3) <<
        values1[0], values1[1], values1[2],
        values1[4], values1[5], values1[6],
        values1[8], values1[9], values1[10]);

    cv::Mat R_gt2 = (cv::Mat_<float>(3, 3) <<
        values2[0], values2[1], values2[2],
        values2[4], values2[5], values2[6],
        values2[8], values2[9], values2[10]);

    R_gt1.copyTo(T_i(cv::Rect(0, 0, 3, 3)));
    GT_pos1.copyTo(T_i(cv::Rect(3, 0, 1, 3)));
    R_gt2.copyTo(T_j(cv::Rect(0, 0, 3, 3)));
    GT_pos2.copyTo(T_j(cv::Rect(3, 0, 1, 3)));

    cv::Mat T_rel_gt = T_j * T_i.inv();
    cv::Mat R_GT = T_rel_gt(cv::Rect(0, 0, 3, 3)).clone();
    cv::Mat t_GT = T_rel_gt(cv::Rect(3, 0, 1, 3)).clone();

    // cv::Mat T_BS = (cv::Mat_<double>(4, 4) <<
    //     0.0148655, -0.999881,  0.0041403, -0.0216401,
    //     0.999557,   0.0149672, 0.0257155, -0.064677,
    //    -0.0257744,  0.0037562, 0.999661,   0.00981073,
    //     0.0,        0.0,       0.0,        1.0);

    double gt_scale = cv::norm(t_GT);
    std::cout << "GT t norm: " << gt_scale << std::endl;

    // Previous pose
    cv::Mat R_prev = R_gt1;
    cv::Mat t_prev = GT_pos1;

    cv::Mat E = K.t() * F * K;

    cv::Mat R_cv, t_cv;
    cv::recoverPose(E, inlierPoints1, inlierPoints2, K, R_cv, t_cv);

    auto [R_world, t_world] = poseEstimator.getPose(F, inlierPoints1, inlierPoints2, gt_scale);

    cv::Vec3d euler_estimated = rotationMatrixToEulerAngles(R_world);
    std::cout << "Estimated angles:" << std::endl;
    std::cout << "  Roll  (X): " << euler_estimated[0] * 180.0 / CV_PI << "°" << std::endl;
    std::cout << "  Pitch (Y): " << euler_estimated[1] * 180.0 / CV_PI << "°" << std::endl;
    std::cout << "  Yaw   (Z): " << euler_estimated[2] * 180.0 / CV_PI << "°" << std::endl;
    // cv::Mat R_GT_rel_body = R_SB * R_GT_rel * R_SB.t();

    std::cout << "Estimated Rotation (world frame):\n" << R_world << "\n";
    std::cout << "Estimated Translation (world frame):\n" << t_world << "\n";

    std::cout << "CV translation: \n" << t_cv << std::endl;

    std::cout << "GT translation (data): " << t_GT << "\n";

    std::cout << "GT rotation: " << R_GT << "\n";

    // Calculating current pose from te relative one
    cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
    R_world.copyTo(T_rel(cv::Rect(0, 0, 3, 3)));
    t_world.copyTo(T_rel(cv::Rect(3, 0, 1, 3)));

    // --- Chain pose: T_j = T_i * T_i->j
    cv::Mat T_estimated = T_i * T_rel.inv();

    std:: cout << "Estimated pose matrix: " << T_estimated << std::endl;
    std::cout << "True pose matrix: " << T_j << std::endl;

    // cv::Mat R_kitti_to_cam = (cv::Mat_<double>(3,3) <<
    //  0,  0, 1,
    // -1,  0, 0,
    //  0, -1, 0);

    // cv::Mat R_adjusted = R_kitti_to_cam * R_GT * R_kitti_to_cam.t();
    cv::Vec3d euler = rotationMatrixToEulerAngles(R_GT);

    std::cout << "Euler angles GT:" << std::endl;
    std::cout << "  Roll  (X): " << euler[0] * 180.0 / CV_PI << "°" << std::endl;
    std::cout << "  Pitch (Y): " << euler[1] * 180.0 / CV_PI << "°" << std::endl;
    std::cout << "  Yaw   (Z): " << euler[2] * 180.0 / CV_PI << "°" << std::endl;

    cv::Vec3d euler_cv = rotationMatrixToEulerAngles(R_cv);
    std::cout << "Euler angles CV translation:" << std::endl;
    std::cout << "  Roll  (X): " << euler_cv[0] * 180.0 / CV_PI << "°" << std::endl;
    std::cout << "  Pitch (Y): " << euler_cv[1] * 180.0 / CV_PI << "°" << std::endl;
    std::cout << "  Yaw   (Z): " << euler_cv[2] * 180.0 / CV_PI << "°" << std::endl;

    cv::Mat R_error = R_GT * R_world.t();
    cv::Vec3d rvec;
    cv::Rodrigues(R_error, rvec);
    double angle_deg = norm(rvec) * 180.0 / CV_PI;
    std::cout << "Rotation error (deg): " << angle_deg << std::endl;

    cv::Mat t_new_norm, t_GT_norm;
    cv::normalize(t_world, t_new_norm);
    cv::normalize(t_GT, t_GT_norm);
    double dot = t_new_norm.dot(t_GT_norm);

    std::cout << "Dot product between estimated and GT (not transformed) direction: " << dot << std::endl;
}

int main()
{
    PoseUpdate poseEstimator;
    testPoseEstimation(poseEstimator);
    return 0;
}


