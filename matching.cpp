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
    float ratioThreshold = 0.75f)
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
    
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

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
    
    std::vector<std::pair<int, int>> matches = matchFloatKeypoints(floatKeypoints1, floatKeypoints2);
    
    std::cout << "Number of matches: " << matches.size() << std::endl;

    const float MATCH_THRESHOLD = 0.75;
    std::cout<<static_cast<float>(matches.size()) / floatKeypoints1.size()<<std::endl;
    if (static_cast<float>(matches.size()) / floatKeypoints1.size() >= MATCH_THRESHOLD) {
        std::cout << "Potentially one object detected!" << std::endl;
    } else {
        std::cout << "Not enough matches to confirm a single object." << std::endl;
    }

    std::vector<cv::DMatch> opencvMatches;
    for (size_t i = 0; i < matches.size(); ++i) {
        opencvMatches.emplace_back(cv::DMatch(matches[i].first, matches[i].second, 0));
    }

    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, opencvMatches, imgMatches);
    
    cv::imshow("Matched Keypoints", imgMatches);
    cv::waitKey(0);
    return 0;
}
