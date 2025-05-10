//
// Created by admin on 04.05.2025.
//

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "PoseUpdate.hpp"

using namespace std;
using namespace cv;

cv::Mat flipZ = (cv::Mat_<double>(4, 4) <<
             1,  0,  0, 0,
             0,  1,  0, 0,
             0,  0, -1, 0,
             0,  0,  0, 1);

cv::Mat readGTLine(const std::string &line) {
    std::stringstream ss(line);
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    for (int i = 0; i < 12; ++i) {
        ss >> T.at<double>(i / 4, i % 4);
    }
    return T;
}

void writePoseCSV(const std::string &filename, const std::vector<cv::Mat> &poses) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open output CSV file.\n";
        return;
    }

    for (const auto &T : poses) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                file << std::setprecision(9) << T.at<double>(r, c);
                if (!(r == 2 && c == 3)) file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

int main() {
    std::string image_dir = "../image_0_5/";   // path to image folder
    std::string pose_file = "../05.txt"; // path to GT poses
    std::string output_csv = "../estimated_poses_5_1_point_4.csv";

    std::ifstream infile(pose_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open pose file.\n";
        return 1;
    }

    std::vector<cv::Mat> gt_poses;
    std::string line;
    while (std::getline(infile, line)) {
        gt_poses.push_back(readGTLine(line));
    }
    infile.close();

    PoseUpdate estimator;
    std::vector<cv::Mat> estimated_poses;
    cv::Mat T_curr = cv::Mat::eye(4, 4, CV_64F);  // Initial pose (identity)
    estimated_poses.push_back(T_curr.clone());

    int skipped_frames = 0;
    size_t i = 1;
    size_t last_valid_frame = 0;

    while (i < gt_poses.size()) {
        std::stringstream ss1, ss2;
        ss1 << std::setw(6) << std::setfill('0') << (last_valid_frame);
        ss2 << std::setw(6) << std::setfill('0') << i;

        std::string img1_path = image_dir + ss1.str() + ".png";
        std::string img2_path = image_dir + ss2.str() + ".png";

        cv::Mat img1 = cv::imread(img1_path);
        cv::Mat img2 = cv::imread(img2_path);
        if (img1.empty() || img2.empty()) {
            std::cerr << "Failed to load images: " << img1_path << " or " << img2_path << "\n";
            estimated_poses.push_back(T_curr.clone());
            ++i;
            continue;
        }

        // --- Feature detection & matching ---
        cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
        std::vector<cv::KeyPoint> kpts1, kpts2;
        cv::Mat desc1, desc2;
        detector->detect(img1, kpts1);
        detector->detect(img2, kpts2);

        cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
        freak->compute(img1, kpts1, desc1);
        freak->compute(img2, kpts2, desc2);

        std::vector<std::vector<cv::DMatch>> knn_matches;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.knnMatch(desc1, desc2, knn_matches, 2);

        std::vector<cv::DMatch> matches;
        for (auto &m : knn_matches) {
            if (m[0].distance < 0.75 * m[1].distance)
                matches.push_back(m[0]);
        }

        if (matches.size() < 8) {
            std::cerr << "Too few matches at frame " << i << "\n";
            cv::Mat T_curr_flipped = flipZ * T_curr;
            estimated_poses.push_back(T_curr_flipped.clone());
            ++skipped_frames;
            ++i;
            continue;
        }

        std::vector<cv::Point2f> pts1, pts2;
        for (auto &m : matches) {
            pts1.push_back(kpts1[m.queryIdx].pt);
            pts2.push_back(kpts2[m.trainIdx].pt);
        }

        std::vector<uchar> inliers;
        cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 1, 0.99, inliers);
        if (F.empty()) {
            std::cerr << "Fundamental matrix failed at frame " << i << "\n";
            estimated_poses.push_back(T_curr.clone());
            ++skipped_frames;
            ++i;
            continue;
        }

        int inlier_count = std::count(inliers.begin(), inliers.end(), 1);
        double inlier_ratio = static_cast<double>(inlier_count) / matches.size();

        std::ofstream hist("inlier_ratios.txt", std::ios::app);
        hist << inlier_ratio << "\n";

        if (inlier_ratio < 0.75 && skipped_frames < 3) {
            std::cout << "Skipping frame " << i << " due to low inlier ratio: " << inlier_ratio << "\n";
            cv::Mat T_curr_flipped = flipZ * T_curr;
            estimated_poses.push_back(T_curr_flipped.clone());
            ++skipped_frames;
            ++i;
            continue;
        }

        skipped_frames = 0;  // reset

        std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
        for (size_t j = 0; j < inliers.size(); ++j) {
            if (inliers[j]) {
                inlier_pts1.push_back(pts1[j]);
                inlier_pts2.push_back(pts2[j]);
            }
        }

        cv::Mat T_rel_gt = gt_poses[i] - gt_poses[last_valid_frame];
        double gt_scale = cv::norm(T_rel_gt(cv::Rect(3, 0, 1, 3)));
        // double gt_scale = (i - last_valid_frame)/1.4;
        std::cout << gt_scale << std::endl;
        last_valid_frame = i;

        auto [R_rel, t_rel] = estimator.getPose(F, inlier_pts1, inlier_pts2, gt_scale);

        cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
        R_rel.copyTo(T_rel(cv::Rect(0, 0, 3, 3)));
        t_rel.copyTo(T_rel(cv::Rect(3, 0, 1, 3)));

        T_curr = T_curr * T_rel;

        cv::Mat T_curr_flipped = flipZ * T_curr;
        estimated_poses.push_back(T_curr_flipped.clone());

        ++i;
    }

    writePoseCSV(output_csv, estimated_poses);
    std::cout << "Wrote estimated poses to: " << output_csv << "\n";

    return 0;
}

