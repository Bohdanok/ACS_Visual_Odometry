//
// Created by julfy1 on 4/1/25.
//

#include "feature_extraction_parallel_GPU.h"

#include <iostream>
#include <opencv2/highgui.hpp>

// #define VISUALIZATION
// #define INTERMEDIATE_TIME_MEASUREMENTS
// #define INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK

void print_descriptor(const std::vector<std::vector<uint8_t>>& descriptor){
    std::cout << "/////////////////////////////////////////////////////////////" << std::endl;
    for (size_t i = 0; i < descriptor.size(); i++) {
        std::cout << "<";
        for (size_t j = 0; j < descriptor[0].size(); j++) {
            std::cout << static_cast<int>(descriptor[i][j]) << ", ";
        }
        std::cout << ">" << std::endl;

    }
    std::cout << "/////////////////////////////////////////////////////////////" << std::endl;

}

bool is_number(const std::string& s) {
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void draw_score_distribution(const std::vector<std::vector<float>>& R_values, const std::string& win_name) {

    int rows = R_values.size();
    int cols = R_values[0].size();

    cv::Mat mat(rows, cols, CV_32F); // Create matrix to store values

    // Copy values
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat.at<float>(i, j) = R_values[i][j];
        }
    }

    // Normalize values to range [0, 1]
    float minVal, maxVal;
    cv::minMaxLoc(mat, reinterpret_cast<double*>(&minVal), reinterpret_cast<double*>(&maxVal));
    std::cout << "Min R: " << minVal << ", Max R: " << maxVal << std::endl;

    // cv::Mat normMat = (mat - minVal) / (maxVal - minVal); // Normalize between 0-1

    float visMax = 300000; // Experiment with this
    cv::Mat normMat = mat / visMax;
    cv::threshold(normMat, normMat, 1.0, 1.0, cv::THRESH_TRUNC);

    // Create color image
    cv::Mat colorImage(rows, cols, CV_8UC3);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = normMat.at<float>(i, j); // Normalized value [0, 1]

            // Map to RGB colors (Green → Blue → Red)
            uchar red   = static_cast<uchar>(255 * std::max(0.f, (val - 0.5f) * 2));
            uchar blue  = static_cast<uchar>(255 * std::max(0.f, (0.5f - std::abs(val - 0.5f)) * 2));
            uchar green = static_cast<uchar>(255 * std::max(0.f, (0.5f - val) * 2));

            colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        }
    }

    cv::imshow(win_name, colorImage);
    // cv::imwrite("../test_images/output_images/" + win_name + ".png", colorImage);
}


cl::Program create_platform_from_binary(const std::string &binary_filename) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();

    std::ifstream bin_file(binary_filename, std::ios::binary | std::ios::ate);
    if (!bin_file.is_open()) {
        throw std::runtime_error("Failed to open kernel binary file: " + binary_filename);
    }

    std::streamsize size = bin_file.tellg();
    bin_file.seekg(0, std::ios::beg);
    std::vector<unsigned char> binary(size);
    if (!bin_file.read(reinterpret_cast<char*>(binary.data()), size)) {
        throw std::runtime_error("Failed to read kernel binary file: " + binary_filename);
    }

    cl::Context context(device);
    cl::Program::Binaries binaries = { binary };
    cl::Program program(context, {device}, binaries);

    program.build({device});

    // try {
    //     program.build({device});
    // } catch (const cl::Error& err) {
    //     std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    //     std::cerr << "Build Log for device " << device.getInfo<CL_DEVICE_NAME>() << ":\n";
    //     std::cerr << buildLog << "\n";
    //     throw;
    // }

    return program;
}


std::vector<std::vector<uint8_t>> feature_extraction_manager(const cv::Mat& image, const GPU_settings& GPU_settings) {
    // Preprocess the image and prepare the enviroment

    // const std::string kernel_path = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/kernels/feature_extraction_kernel_functions.bin"; // TODO argv?

    // const std::string image_filename = "/home/julfy/Documents/ACS/ACS_Visual_Odometry/images/Zhovkva2.jpg";

    // const auto program = create_platform_from_binary(kerlen_filename);
    //
    // const auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    // const auto& device = devices.front(); // TODO Take the best device
    // const auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
    //
    // const GPU_settings GPU_settings({program, device, context});

    cv::Mat my_blurred_gray;
    cv::GaussianBlur(image, my_blurred_gray, cv::Size(7, 7), 0);

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;

    std::vector<std::vector<float>> R_score(n_rows, std::vector<float>(n_cols));

#ifdef INTERMEDIATE_TIME_MEASUREMENTS
    const auto start = get_current_time_fenced();
#endif

    CornerDetectionParallel_GPU::shitomasi_corner_detection(GPU_settings, my_blurred_gray, R_score);

#ifdef INTERMEDIATE_TIME_MEASUREMENTS
    const auto end = get_current_time_fenced();
    std::cout << "GPU response calculations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
#endif

    // draw_score_distribution(R_score, "Response");
    //
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    auto local_mins_shitomasi = CornerDetectionParallel_GPU::non_maximum_suppression(R_score, n_rows, n_cols, 5, 1500);

    std::sort(local_mins_shitomasi.begin(), local_mins_shitomasi.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        const int ay = static_cast<int>(a.pt.y);
        const int by = static_cast<int>(b.pt.y);
        if (ay == by)
            return static_cast<int>(a.pt.x) < static_cast<int>(b.pt.x);
        return ay < by;
    });

#ifdef VISUALIZATION
        for (const auto coords : local_mins_shitomasi) {

            // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

            cv::circle(image, cv::Point(coords.pt.x, coords.pt.y), 5, cv::Scalar(0, 0, 255), 3);
        }
    draw_score_distribution(R_score, "Response");

    cv::imwrite("COrners.jpg", image);
    cv::imshow("Corners GPU", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
#endif
    // std::vector<std::vector<uint8_t>> descriptor = {{1}};

    auto descriptor = FREAK_Parallel_GPU::FREAK_feature_description(local_mins_shitomasi, my_blurred_gray, GPU_settings);

    // print_descriptor(descriptor);

    return descriptor;
}


std::pair<std::vector<std::vector<uint8_t>>, std::vector<cv::KeyPoint>> feature_extraction_manager_with_points(const cv::Mat& image, const GPU_settings& GPU_settings) {
// #ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
//     const auto start_gaussian = get_current_time_fenced();
// #endif

    cv::Mat my_blurred_gray;
    cv::GaussianBlur(image, my_blurred_gray, cv::Size(7, 7), 0);

    const int n_rows = my_blurred_gray.rows;
    const int n_cols = my_blurred_gray.cols;

    std::vector<std::vector<float>> R_score(n_rows, std::vector<float>(n_cols));


// #ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
//     const auto end_gaussian = get_current_time_fenced();
//     std::cout << "Gaussian calculation:: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gaussian - start_gaussian).count()
//               << " ms" << std::endl;
// #endif
//


#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto start = get_current_time_fenced();
#endif

    CornerDetectionParallel_GPU::shitomasi_corner_detection(GPU_settings, my_blurred_gray, R_score);

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end = get_current_time_fenced();
    std::cout << "GPU response calculations: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
#endif

    // draw_score_distribution(R_score, "Response");
    //
    // cv::waitKey(0);
    // cv::destroyAllWindows();
// #ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
//     const auto start_nms = get_current_time_fenced();
// #endif
    auto local_mins_shitomasi = CornerDetectionParallel_GPU::non_maximum_suppression(R_score, n_rows, n_cols, 3, 2000);
    // cv::Mat new_image;
    // if (image.channels() == 1) cv::cvtColor(image, new_image, cv::COLOR_GRAY2BGR);
    //
    // for (auto coords : local_mins_shitomasi) {
    //
    //     // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;
    //
    //     cv::circle(new_image, cv::Point(coords.pt.x, coords.pt.y), 1, cv::Scalar(0, 0, 255), 3);
    // }
    // cv::imshow("GPU CORNERS", new_image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

// #ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
//     const auto end_nms = get_current_time_fenced();
//     std::cout << "Non maximum suppression: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_nms - start_nms).count()
//               << " ms" << std::endl;
// #endif


// #ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
//     const auto start_sorting = get_current_time_fenced();
// #endif
    std::sort(local_mins_shitomasi.begin(), local_mins_shitomasi.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        const int ay = static_cast<int>(a.pt.y);
        const int by = static_cast<int>(b.pt.y);
        if (ay == by)
            return static_cast<int>(a.pt.x) < static_cast<int>(b.pt.x);
        return ay < by;
    });

// #ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
//     const auto end_sorting = get_current_time_fenced();
//     std::cout << "Sorting time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_sorting - start_sorting).count()
//               << " ms" << std::endl;
// #endif



#ifdef VISUALIZATION
        for (const auto coords : local_mins_shitomasi) {

            // std::cout << "(" << std::get<0>(coords) << ", " << std::get<1>(coords) << ")" << std::endl;

            cv::circle(image, cv::Point(coords.pt.x, coords.pt.y), 5, cv::Scalar(0, 0, 255), 3);
        }
    draw_score_distribution(R_score, "Response");

    cv::imwrite("COrners.jpg", image);
    cv::imshow("Corners GPU", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
#endif
    // std::vector<std::vector<uint8_t>> descriptor = {{1}};
#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto buffer_read_start = get_current_time_fenced();
#endif

    auto descriptor = FREAK_Parallel_GPU::FREAK_feature_description(local_mins_shitomasi, my_blurred_gray, GPU_settings);

#ifdef INTERMEDIATE_TIME_MEASUREMENTS_GPU_WORK
    const auto end_buffer_read = get_current_time_fenced();
    // buffer_write_time += std::chrono::duration_cast<std::chrono::milliseconds>(start_buffer_write - end_buffer_write).count();
    std::cout << "FEATURE EXTRACTION DESCRIPTOR FUNCTION: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_buffer_read - buffer_read_start).count()
              << " ms" << std::endl;
#endif

    // print_descriptor(descriptor);

    return {descriptor, local_mins_shitomasi};
}
