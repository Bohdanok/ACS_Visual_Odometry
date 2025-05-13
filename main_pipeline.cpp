//
// Created by admin on 10.05.2025.
//

#include "VisualOdometry.h"

int main() {
    // std::string kernel_file = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/kernels/feature_extraction_kernel_functions.bin";
    // std::size_t num_threads = 8;
    //
    // std::string image_dir = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/data/data/sequences/images_5/";
    // std::size_t num_images = 500;
    // std::string pose_file = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/data/data/poses/05.txt";
    // std::string output_csv = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/estimated_poses_our_5.csv";

    std::string kernel_file = "../kernels/feature_extraction_kernel_functions.bin";
    std::size_t num_threads = 16;

    std::string image_dir = "../images_5/";
    std::size_t num_images = 10;
    std::string pose_file = "../05.txt";
    std::string output_csv = "../estimated_poses_our.csv";

    auto start = get_current_time_fenced();

    VisualOdometry vo(kernel_file, num_threads);

    vo.run(image_dir, num_images, pose_file, output_csv);

    auto end = get_current_time_fenced();

    std::cout << "Time for the dataset pose estimation: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms for " << num_images << " images" << std::endl;

    return 0;
}
