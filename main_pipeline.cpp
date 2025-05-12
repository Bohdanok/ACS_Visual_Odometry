//
// Created by admin on 10.05.2025.
//

#include "VisualOdometry.h"

int main() {
    std::string kernel_file = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/kernels/feature_extraction_kernel_functions.bin";
    std::size_t num_threads = 8;

    std::string image_dir = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/data/data/sequences/images_5/";
    std::size_t num_images = 500;
    std::string pose_file = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/data/data/poses/05.txt";
    std::string output_csv = "/mnt/d/pok/project_directory/ACS_Visual_Odometry/estimated_poses_our_5.csv";

    VisualOdometry vo(kernel_file, num_threads);

    vo.run(image_dir, num_images, pose_file, output_csv);

    return 0;
}
