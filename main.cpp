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

 using namespace std;
 using namespace cv;

 cv::Mat readGTLine(const std::string &line) {
     std::stringstream ss(line);
     cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
     for (int i = 0; i < 12; ++i) {
         ss >> T.at<double>(i / 4, i % 4);
     }
     return T;
 }

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
 {
     std::atomic_thread_fence(std::memory_order_seq_cst);
     auto res_time = std::chrono::high_resolution_clock::now();
     std::atomic_thread_fence(std::memory_order_seq_cst);
     return res_time;
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

std::vector<cv::KeyPoint> convertToKeypoints(const std::vector<cv::Point2f>& corners) {
     std::vector<cv::KeyPoint> keypoints;
     for (const auto& pt : corners) {
         keypoints.emplace_back(pt, 1.f);  // size = 1.0f
     }
     return keypoints;
 }

 int main() {
     // std::string image_dir = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/data/data/sequences/images_5/";
     // // std::size_t num_images = 500;
     // std::string pose_file = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/data/data/poses/05.txt";
     // std::string output_csv = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/estimated_poses_opencv_5.csv";

     std::string kernel_file = "../kernels/feature_extraction_kernel_functions.bin";
     std::size_t num_threads = 16;

     std::string image_dir = "../images_5/";
     std::size_t num_images = 1500;
     std::string pose_file = "../05.txt";
     std::string output_csv = "../estimated_poses_our.csv";

     auto start = get_current_time_fenced();


     const double fx = 7.188560000000e+02;
     const double fy = 7.188560000000e+02;
     const double cx = 6.071928000000e+02;
     const double cy = 1.852157000000e+02;
     Mat K = (Mat_<double>(3,3) << fx, 0, cx,
                                    0, fy, cy,
                                    0,  0,  1);

     Mat flipZ = Mat::eye(4,4,CV_64F);
     flipZ.at<double>(2,2) = -1;

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

     std::vector<cv::Mat> estimated_poses;
     cv::Mat T_curr = cv::Mat::eye(4, 4, CV_64F);  // Initial pose (identity)
     estimated_poses.push_back(T_curr.clone());

     int skipped_frames = 0;
     size_t i = 1;
     size_t last_valid_frame = 0;

     cv::Mat img1 = cv::imread(image_dir + "000000.png", cv::IMREAD_GRAYSCALE);
     if (img1.empty()) {
         std::cerr << "Failed to load first image: " << image_dir + "000000.png\n";
         return -1;
     }

     int maxCorners = 1500;
     double qualityLevel = 0.01;
     double minDistance = 10.0;

     std::vector<cv::KeyPoint> kpts1, kpts2;
     cv::Mat desc1, desc2;

     std::vector<cv::Point2f> corners1, corners2;
     cv::goodFeaturesToTrack(cv::imread(image_dir + "000000.png", cv::IMREAD_GRAYSCALE), corners1, maxCorners, qualityLevel, minDistance);

     kpts1 = convertToKeypoints(corners1);

     cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();

     freak->compute(img1, kpts1, desc1);

     while (i < num_images) {
         // std::cout << "Processing frame " << i << std::endl;
         std::stringstream ss1, ss2;
//         ss1 << std::setw(6) << std::setfill('0') << (last_valid_frame);
         ss2 << std::setw(6) << std::setfill('0') << i;

//         std::string img1_path = image_dir + ss1.str() + ".png";
         std::string img2_path = image_dir + ss2.str() + ".png";

         cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
         if (img2.empty()) {
             std::cerr << "Failed to load image: " << img2_path << "\n";
             estimated_poses.push_back(T_curr.clone());
             ++i;
             continue;
         }

         cv::goodFeaturesToTrack(img2, corners2, maxCorners, qualityLevel, minDistance);

         kpts2 = convertToKeypoints(corners2);

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

//         std::ofstream hist("inlier_ratios.txt", std::ios::app);
//         hist << inlier_ratio << "\n";

         if (inlier_ratio < 0.5 && skipped_frames < 3) {
             // std::cout << "Skipping frame " << i << " due to low inlier ratio: " << inlier_ratio << "\n";
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

         cv::Mat T_rel_gt = gt_poses[i].inv() * gt_poses[last_valid_frame];
         double gt_scale = cv::norm(T_rel_gt(cv::Rect(3, 0, 1, 3)));
         last_valid_frame = i;
         desc1 = std::move(desc2);
         kpts1 = std::move(kpts2);

         vector<uchar> maskE;
         Mat E = findEssentialMat(inlier_pts1, inlier_pts2, K, RANSAC, 0.99, 1.0, maskE);
         if (E.empty()) {
             cerr << "Не вдалось знайти E у кадрі " << i << endl;
             estimated_poses.push_back(flipZ * T_curr);
             skipped_frames++;
             ++i;
             continue;
         }
         Mat R, t;
         recoverPose(E, inlier_pts1, inlier_pts2, K, R, t, maskE);

         t *= gt_scale;

         cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
         R.copyTo(T_rel(cv::Rect(0, 0, 3, 3)));
         t.copyTo(T_rel(cv::Rect(3, 0, 1, 3)));

         T_curr = T_curr * T_rel;

         cv::Mat T_curr_flipped = flipZ * T_curr;
         estimated_poses.push_back(T_curr_flipped.clone());

         ++i;
     }

     writePoseCSV(output_csv, estimated_poses);
     std::cout << "Wrote estimated poses to: " << output_csv << "\n";

     auto end = get_current_time_fenced();

     std::cout << "Time for the dataset pose estimation: "
               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
               << " ms for " << num_images << " images" << std::endl;

     return 0;
 }

