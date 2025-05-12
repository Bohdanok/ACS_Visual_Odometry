//
// Created by julfy on 20.04.25.
//

#include "VisualOdometry.h"

VisualOdometry::VisualOdometry(
    const std::string& kernel_filename,
    const std::size_t num_threads)
  : VO_pool(num_threads)
  , number_of_threads(num_threads)
  , gpu_settings([&]() {
      const auto program = create_platform_from_binary(kernel_filename);
      const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
      if (devices.empty()) {
          throw std::runtime_error("No OpenCL devices found for the program.");
      }
      const auto& device = devices.front();
      const auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
      return GPU_settings{program, device, context};
  }())
  , kernel_filename(kernel_filename)
{   }


// READ in grayscale already
std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> VisualOdometry::compute_descriptor_with_key_points(const cv::Mat &image) {
    return feature_extraction_manager_with_points(image, this->gpu_settings);
}


std::vector<std::pair<int, int>> VisualOdometry::match_descriptors(const std::vector<std::vector<uint8_t>> &desc1, const std::vector<std::vector<uint8_t>> &desc2) {
    return matchCustomBinaryDescriptorsThreadPool(desc1, desc2, this->VO_pool, this->number_of_threads, 0.75f);
}

void VisualOdometry::run(const std::string image_dir, const size_t num_images, const std::string pose_file, const std::string output_csv)
{
    std::ifstream infile(pose_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open pose file.\n";
        return;
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

    auto [desc1, kpts1] = compute_descriptor_with_key_points(
        cv::imread(image_dir + "000000.png", cv::IMREAD_GRAYSCALE)
    );


    while (i < gt_poses.size()) {
        std::stringstream ss1, ss2;

        std::cout << "Frame number: " << i << std::endl;
        // ss1 << std::setw(6) << std::setfill('0') << (last_valid_frame);
        ss2 << std::setw(6) << std::setfill('0') << i;

        // std::string img1_path = image_dir + ss1.str() + ".png";
        std::string img2_path = image_dir + ss2.str() + ".png";

        // cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
        if (img2.empty()) {
            std::cerr << "Failed to load image: " << img2_path << "\n";
            estimated_poses.push_back(T_curr.clone());
            ++i;
            continue;
        }

        // auto [desc1, kpts1] = this->compute_descriptor_with_key_points(img1);
        auto [desc2, kpts2] = this->compute_descriptor_with_key_points(img2);

        std::vector<std::pair<int, int>> match_indices = this->match_descriptors(desc1, desc2);

        if (match_indices.size() < 8)
        {
            std::cerr << "Too few matches at frame " << i << "\n";
            cv::Mat T_curr_flipped = flipZ * T_curr;
            estimated_poses.push_back(T_curr_flipped.clone());
            ++skipped_frames;
            ++i;
            continue;
        }

        std::cout << "Matched indexes number: " << match_indices.size() << std::endl;

        std::vector<cv::Point2f> pts1, pts2;
        for (auto &[idx1, idx2] : match_indices) {
            pts1.push_back(kpts1[idx1].pt);
            pts2.push_back(kpts2[idx2].pt);
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
        std::cout << "Number of inliers: " << inlier_count << "\n" << std::endl;
        double inlier_ratio = static_cast<double>(inlier_count) / match_indices.size();

        std::ofstream hist("inlier_ratios.txt", std::ios::app);
        hist << inlier_ratio << "\n";

        if (inlier_ratio < 0.5 && skipped_frames < 3) {
            std::cout << "Skipping frame " << i << " due to low inlier ratio: " << inlier_ratio << "\n";
            cv::Mat T_curr_flipped = flipZ * T_curr;
            estimated_poses.push_back(T_curr_flipped.clone());
            ++skipped_frames;
            ++i;
            continue;
        }

        skipped_frames = 0;

        std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
        for (size_t j = 0; j < inliers.size(); ++j) {
            if (inliers[j]) {
                inlier_pts1.push_back(pts1[j]);
                inlier_pts2.push_back(pts2[j]);
            }
        }

        // std::cout << gt_poses[i] << std::endl;
        cv::Mat T_rel_gt = gt_poses[i].inv() * gt_poses[last_valid_frame];
        double gt_scale = cv::norm(T_rel_gt(cv::Rect(3, 0, 1, 3)));
        // double gt_scale = (i - last_valid_frame)/1.4;
        // std::cout << gt_scale << std::endl;

        // cv::Mat T_i = utils::toHomogeneous(gt_poses[last_valid_frame]);
        // cv::Mat T_j = utils::toHomogeneous(gt_poses[i]);
        // cv::Mat T_rel_ = T_i.inv() * T_j;
        //
        // cv::Mat t_vec = T_rel_(cv::Range(0,3), cv::Range(3,4));  // 3×1
        // double gt_scale = cv::norm(t_vec);

        last_valid_frame = i;
        desc1 = std::move(desc2);
        kpts1 = std::move(kpts2);

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
}
