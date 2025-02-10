#include <iostream>
#include <vector>
#include <bitset>
#include <limits>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

const int BINARY_DESCRIPTOR_SIZE = 32;
const int FLOAT_DESCRIPTOR_SIZE = 4;

struct BinaryKeypoint {
    int x, y;
    uint32_t descriptor;
};

struct FloatKeypoint {
    int x, y;
    float descriptor[FLOAT_DESCRIPTOR_SIZE];
};

int hammingDistance(uint32_t d1, uint32_t d2) {
    return std::bitset<32>(d1 ^ d2).count();
}

float euclideanDistance(const float* d1, const float* d2) {
    float sum = 0.0f;
    for (int i = 0; i < FLOAT_DESCRIPTOR_SIZE; ++i) {
        float diff = d1[i] - d2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<std::pair<int, int> > matchBinaryKeypoints(
    const std::vector<BinaryKeypoint>& keypoints1,
    const std::vector<BinaryKeypoint>& keypoints2,
    float ratioThreshold = 0.75f)
{
    std::vector<std::pair<int, int> > matches;
    for (size_t i = 0; i < keypoints1.size(); ++i) {
        int bestIdx = -1, secondBestIdx = -1;
        int bestDist = std::numeric_limits<int>::max();
        int secondBestDist = std::numeric_limits<int>::max();

        for (size_t j = 0; j < keypoints2.size(); ++j) {
            int dist = hammingDistance(keypoints1[i].descriptor, keypoints2[j].descriptor);
            if (dist < bestDist) {
                secondBestDist = bestDist;
                secondBestIdx = bestIdx;
                bestDist = dist;
                bestIdx = j;
            } else if (dist < secondBestDist) {
                secondBestDist = dist;
                secondBestIdx = j;
            }
        }

        if (bestIdx != -1 && secondBestIdx != -1 && bestDist < ratioThreshold * secondBestDist) {
            matches.emplace_back(i, bestIdx);
        }
    }
    return matches;
}

std::vector<std::pair<int, int> > matchFloatKeypoints(
    const std::vector<FloatKeypoint>& keypoints1,
    const std::vector<FloatKeypoint>& keypoints2,
    float ratioThreshold = 0.75)
{
    std::vector<std::pair<int, int> > matches;
    for (size_t i = 0; i < keypoints1.size(); ++i) {
        int bestIdx = -1, secondBestIdx = -1;
        float bestDist = std::numeric_limits<float>::max();
        float secondBestDist = std::numeric_limits<float>::max();
        // std::cout<<1<<std::endl<<keypoints1[i].x<<keypoints1[i].y;
        for (size_t j = 0; j < keypoints2.size(); ++j) {
            float dist = euclideanDistance(keypoints1[i].descriptor, keypoints2[j].descriptor);
            if (dist < bestDist) {
                secondBestDist = bestDist;
                secondBestIdx = bestIdx;
                bestDist = dist;
                bestIdx = j;
            } else if (dist < secondBestDist) {
                secondBestDist = dist;
                secondBestIdx = j;
            }
        }

        if (bestIdx != -1 && secondBestIdx != -1 && bestDist < ratioThreshold * secondBestDist) {
            matches.emplace_back(i, bestIdx);
        }
    }
    return matches;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./matching <image1> <image2>" << std::endl;
        return -1;
    }
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    
    if (img1.empty() || img2.empty()) {
        std::cout << "Error loading images!" << std::endl;
        return -1;
    }

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    sift->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

    std::vector<FloatKeypoint> floatKeypoints1, floatKeypoints2;
    for (size_t i = 0; i < keypoints1.size(); ++i) {
        FloatKeypoint kp;
        kp.x = keypoints1[i].pt.x;
        kp.y = keypoints1[i].pt.y;
        for (int j = 0; j < FLOAT_DESCRIPTOR_SIZE; ++j) {
            kp.descriptor[j] = descriptors1.at<float>(i, j);
        }
        floatKeypoints1.push_back(kp);
    }
    
    for (size_t i = 0; i < keypoints2.size(); ++i) {
        FloatKeypoint kp;
        kp.x = keypoints2[i].pt.x;
        kp.y = keypoints2[i].pt.y;
        for (int j = 0; j < FLOAT_DESCRIPTOR_SIZE; ++j) {
            kp.descriptor[j] = descriptors2.at<float>(i, j);
        }
        floatKeypoints2.push_back(kp);
    }
    float MATCH_THRESHOLD = 0.5;
    std::vector<std::pair<int, int>> floatMatches = matchFloatKeypoints(floatKeypoints1, floatKeypoints2);
    
    std::cout << "Float matches: " << floatMatches.size() << " " << floatKeypoints1.size() << " " << floatKeypoints2.size() << std::endl;
    if (floatMatches.size() / std::min(static_cast<float>(floatKeypoints1.size()), static_cast<float>(floatKeypoints2.size())) >= MATCH_THRESHOLD) {
        std::cout << "Potentially one object detected using float-based descriptors!" << std::endl;
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : floatMatches) {
            points1.push_back(cv::Point2f(floatKeypoints1[match.first].x, floatKeypoints1[match.first].y));
            points2.push_back(cv::Point2f(floatKeypoints2[match.second].x, floatKeypoints2[match.second].y));
        }

        std::vector<uchar> inliers;
        cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0, inliers);

        std::vector<cv::DMatch> ransacMatches;
        for (size_t i = 0; i < floatMatches.size(); ++i) {
            if (inliers[i]) {
                ransacMatches.push_back(cv::DMatch(floatMatches[i].first, floatMatches[i].second, 0));
            }
        }

        cv::Mat imgMatches;
        cv::drawMatches(img1, keypoints1, img2, keypoints2, ransacMatches, imgMatches);
        cv::imshow("RANSAC Filtered Float Keypoint Matches", imgMatches);
    } else {
        std::cout << "Not enough matches to confirm a single object using float-based descriptors." << std::endl;
    }


    // === BRISK для бінарних дескрипторів ===
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    std::vector<cv::KeyPoint> binKeypoints1, binKeypoints2;
    cv::Mat binDescriptors1, binDescriptors2;

    brisk->detectAndCompute(img1, cv::Mat(), binKeypoints1, binDescriptors1);
    brisk->detectAndCompute(img2, cv::Mat(), binKeypoints2, binDescriptors2);

    std::vector<BinaryKeypoint> binaryKeypoints1, binaryKeypoints2;
    for (size_t i = 0; i < binKeypoints1.size(); ++i) {
        BinaryKeypoint kp;
        kp.x = binKeypoints1[i].pt.x;
        kp.y = binKeypoints1[i].pt.y;
        kp.descriptor = binDescriptors1.at<uint32_t>(i, 0);
        binaryKeypoints1.push_back(kp);
    }
    
    for (size_t i = 0; i < binKeypoints2.size(); ++i) {
        BinaryKeypoint kp;
        kp.x = binKeypoints2[i].pt.x;
        kp.y = binKeypoints2[i].pt.y;
        kp.descriptor = binDescriptors2.at<uint32_t>(i, 0);
        binaryKeypoints2.push_back(kp);
    }

    std::vector<std::pair<int, int>> binaryMatches = matchBinaryKeypoints(binaryKeypoints1, binaryKeypoints2);
    
    std::cout << "Binary matches: " << binaryMatches.size() << " " << binaryKeypoints1.size() << " " << binaryKeypoints2.size()<<std::endl;
    if (binaryMatches.size() / std::min(static_cast<float>(binaryKeypoints1.size()), static_cast<float>(binaryKeypoints2.size())) >= MATCH_THRESHOLD) {
        std::cout << "Potentially one object detected using binary descriptors!" << std::endl;
    

        std::vector<cv::Point2f> binaryPoints1, binaryPoints2;
        for (const auto& match : binaryMatches) {
            binaryPoints1.push_back(cv::Point2f(binaryKeypoints1[match.first].x, binaryKeypoints1[match.first].y));
            binaryPoints2.push_back(cv::Point2f(binaryKeypoints2[match.second].x, binaryKeypoints2[match.second].y));
        }

        std::vector<uchar> binaryInliers;
        cv::Mat binaryHomography = cv::findHomography(binaryPoints1, binaryPoints2, cv::RANSAC, 3.0, binaryInliers);

        std::vector<cv::DMatch> binaryRansacMatches;
        for (size_t i = 0; i < binaryMatches.size(); ++i) {
            if (binaryInliers[i]) {
                binaryRansacMatches.push_back(cv::DMatch(binaryMatches[i].first, binaryMatches[i].second, 0));
            }
        }

        cv::Mat binaryImgMatches;
        
        cv::drawMatches(img1, keypoints1, img2, keypoints2, binaryRansacMatches, binaryImgMatches);
        cv::imshow("RANSAC Filtered Binary Keypoint Matches", binaryImgMatches);
    } else {
        std::cout << "Not enough matches to confirm a single object using binary descriptors." << std::endl;
    }

    cv::waitKey(0);
    return 0;
}