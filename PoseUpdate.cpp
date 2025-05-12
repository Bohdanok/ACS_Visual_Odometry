//
// Created by admin on 17.04.2025.
//

#include "PoseUpdate.hpp"

cv::Mat flipZ = (cv::Mat_<double>(4, 4) <<
             1,  0,  0, 0,
             0,  1,  0, 0,
             0,  0, -1, 0,
             0,  0,  0, 1);


cv::Vec3d rotationMatrixToEulerAngles(const cv::Mat &R) {
    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +
                     R.at<double>(1,0) * R.at<double>(1,0));

    bool singular = sy < 1e-6;

    double x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2,1), R.at<double>(2,2)); // roll
        y = atan2(-R.at<double>(2,0), sy);               // pitch
        z = atan2(R.at<double>(1,0), R.at<double>(0,0)); // yaw
    } else {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }

    return cv::Vec3d(x, y, z); // (roll, pitch, yaw)
}

cv::Mat quat_to_rot(double w, double x, double y, double z) {
    cv::Mat R = (cv::Mat_<double>(3,3) <<
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w,
        2 * x * y + 2 * z * w,     1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
        2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w,     1 - 2 * x * x - 2 * y * y
    );
    return R;
}

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
