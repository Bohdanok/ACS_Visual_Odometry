#include "VisualOdometry.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_threads> <image_dir> <num_images> <pose_file> <output_csv>" << std::endl;
        return EXIT_FAILURE;
    }

    std::size_t num_threads = 0;
    try {
        num_threads = static_cast<std::size_t>(std::stoul(argv[1]));
    } catch (const std::exception& e) {
        std::cerr << "Invalid num_threads: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    std::string image_dir = argv[2];
    std::size_t num_images = 0;
    try {
        num_images = static_cast<std::size_t>(std::stoul(argv[3]));
    } catch (const std::exception& e) {
        std::cerr << "Invalid num_images: " << argv[3] << std::endl;
        return EXIT_FAILURE;
    }
    std::string pose_file = argv[4];
    std::string output_csv = argv[5];

    std::filesystem::path kernel_file = "../kernels/feature_extraction_kernel_functions.bin";

    if (!std::filesystem::exists(kernel_file)) {
        std::cerr << "Kernel file not found: " << kernel_file << std::endl;
        return EXIT_FAILURE;
    }

    VisualOdometry vo(kernel_file.string(), num_threads);
    vo.run(image_dir, num_images, pose_file, output_csv);

    return EXIT_SUCCESS;
}
