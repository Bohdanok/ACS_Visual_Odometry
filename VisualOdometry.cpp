//
// Created by julfy on 20.04.25.
//

#include "VisualOdometry.h"

// #define PRINT_INTERMEDIATE_STEPS

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
    Ransac ransac;
    FundamentalMatrix model;
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

    while (i < num_images) {
        std::stringstream ss2;
#ifdef PRINT_INTERMEDIATE_STEPS
        std::cout << "Frame number: " << i << std::endl;
#endif
        ss2 << std::setw(6) << std::setfill('0') << i;
        std::string img2_path = image_dir + ss2.str() + ".png";

        cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
        if (img2.empty()) {
            std::cerr << "Failed to load image: " << img2_path << "\n";
            estimated_poses.push_back(T_curr.clone());
            ++i;
            continue;
        }

        // ==== Feature extraction ====
#ifdef PRINT_INTERMEDIATE_STEPS
        auto t_start = get_current_time_fenced();
#endif
        auto [desc2, kpts2] = this->compute_descriptor_with_key_points(img2);
#ifdef PRINT_INTERMEDIATE_STEPS
        auto t_end = get_current_time_fenced();
        std::cout << "Feature extraction time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
                  << " ms" << std::endl;
#endif

        // ==== Matching ====
#ifdef PRINT_INTERMEDIATE_STEPS
        t_start = get_current_time_fenced();
#endif
        std::vector<std::pair<int, int>> match_indices = this->match_descriptors(desc1, desc2);
#ifdef PRINT_INTERMEDIATE_STEPS
        t_end = get_current_time_fenced();
        std::cout << "Descriptor matching time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
                  << " ms" << std::endl;
#endif

        if (match_indices.size() < 8) {
            std::cerr << "Too few matches at frame " << i << "\n";
            cv::Mat T_curr_flipped = flipZ * T_curr;
            estimated_poses.push_back(T_curr_flipped.clone());
            ++skipped_frames;
            ++i;
            continue;
        }

        std::vector<std::pair<Point, Point>> matchedPoints;
        for (const auto &[idx1, idx2] : match_indices) {
            matchedPoints.emplace_back(
                Point{static_cast<double>(kpts1[idx1].pt.x), static_cast<double>(kpts1[idx1].pt.y)},
                Point{static_cast<double>(kpts2[idx2].pt.x), static_cast<double>(kpts2[idx2].pt.y)}
            );
        }

        // ==== RANSAC ====
#ifdef PRINT_INTERMEDIATE_STEPS
        t_start = get_current_time_fenced();
#endif

        ransac.run(model, matchedPoints, 0.99, 1.0, number_of_threads, this->VO_pool);
        // Ransac::run(model, matchedPoints, 0.99, 1.0, number_of_threads);
        // Ransac::run(model, matchedPoints, 0.99, 1.0, number_of_threads, this->VO_pool);
#ifdef PRINT_INTERMEDIATE_STEPS
        t_end = get_current_time_fenced();
        std::cout << "RANSAC time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
                  << " ms" << std::endl;
#endif

        Eigen::Matrix3d F_eigen = model.getMatrix();
        cv::Mat F(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                F.at<double>(r, c) = F_eigen(r, c);

        const auto& inliers = model.getInliers();
        if (inliers.size() < 8) {
            cv::Mat T_curr_flipped = flipZ * T_curr;
            estimated_poses.push_back(T_curr_flipped.clone());
            ++skipped_frames;
            ++i;
            continue;
        }

        std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
        for (const auto& p : inliers) {
            inlier_pts1.emplace_back(p.first.x, p.first.y);
            inlier_pts2.emplace_back(p.second.x, p.second.y);
        }

        cv::Mat T_rel_gt = gt_poses[i].inv() * gt_poses[last_valid_frame];
        double gt_scale = cv::norm(T_rel_gt(cv::Rect(3, 0, 1, 3)));

        last_valid_frame = i;
        desc1 = std::move(desc2);
        kpts1 = std::move(kpts2);

        // ==== Pose Estimation ====
#ifdef PRINT_INTERMEDIATE_STEPS
        t_start = get_current_time_fenced();
#endif
        auto [R_rel, t_rel] = estimator.getPose(F, inlier_pts1, inlier_pts2, gt_scale);
#ifdef PRINT_INTERMEDIATE_STEPS
        t_end = get_current_time_fenced();
        std::cout << "Pose estimation time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
                  << " ms\n" << std::endl;
#endif

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
