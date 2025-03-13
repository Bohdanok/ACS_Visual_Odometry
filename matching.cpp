#include <iostream>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <bitset>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

const int BINARY_DESCRIPTOR_SIZE = 32;
const double MATCH_THRESHOLD = 0.5;

int hammingDistance(const uint8_t* d1, const uint8_t* d2, int length) {
    int distance = 0;
    int i = 0;

    for (; i + 4 <= length; i += 4) {
        uint32_t v1, v2;
        std::memcpy(&v1, d1 + i, sizeof(uint32_t));
        std::memcpy(&v2, d2 + i, sizeof(uint32_t));
        distance += __builtin_popcount(v1 ^ v2);
    }

    for (; i < length; ++i) {
        distance += __builtin_popcount(d1[i] ^ d2[i]);
    }

    return distance;
}

std::vector<std::pair<int, int>> matchBinaryKeypoints(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    float ratioThreshold = 0.75f)
{
    std::vector<std::pair<int, int>> matches;
    for (int i = 0; i < descriptors1.rows; ++i) {
        int bestIdx = -1, secondBestIdx = -1;
        int bestDist = std::numeric_limits<int>::max();
        int secondBestDist = std::numeric_limits<int>::max();
        const uint8_t* desc1 = descriptors1.ptr<uint8_t>(i);
        for (int j = 0; j < descriptors2.rows; ++j) {
            const uint8_t* desc2 = descriptors2.ptr<uint8_t>(j);
            int dist = hammingDistance(desc1, desc2, descriptors1.cols);
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

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

std::vector<std::pair<int, int>> findCommonMatches(
    const std::vector<std::pair<int, int>>& customMatches,
    const std::vector<cv::DMatch>& cvMatches)
{
    std::unordered_set<std::pair<int, int>, PairHash> cvMatchSet;
    for (const auto& cvm : cvMatches) {
        cvMatchSet.insert({cvm.queryIdx, cvm.trainIdx});
    }
    std::vector<std::pair<int, int>> commonMatches;
    for (const auto& custom : customMatches) {
        if (cvMatchSet.find(custom) != cvMatchSet.end()) {
            commonMatches.push_back(custom);
        }
    }
    return commonMatches;
}

std::vector<cv::DMatch> convertToDMatch(const std::vector<std::pair<int, int>>& matches) {
    std::vector<cv::DMatch> dMatches;
    for (const auto& match : matches) {
        dMatches.emplace_back(cv::DMatch(match.first, match.second, 0));
    }
    return dMatches;
}

std::vector<cv::DMatch> applyRANSAC(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2)
{
    if (matches.size() < 4) return matches;

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(keypoints1[match.queryIdx].pt);
        pts2.push_back(keypoints2[match.trainIdx].pt);
    }

    std::vector<uchar> inliersMask;
    cv::Mat homography = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, inliersMask);

    std::vector<cv::DMatch> filteredMatches;
    for (size_t i = 0; i < matches.size(); i++) {
        if (inliersMask[i]) {
            filteredMatches.push_back(matches[i]);
        }
    }
    return filteredMatches;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./matching <image1> <image2>" << std::endl;
        return -1;
    }

    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error reading images." << std::endl;
        return -1;
    }

    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();

    std::vector<cv::KeyPoint> briskKps1, briskKps2;
    cv::Mat briskDesc1, briskDesc2;

    brisk->detectAndCompute(img1, cv::noArray(), briskKps1, briskDesc1);
    brisk->detectAndCompute(img2, cv::noArray(), briskKps2, briskDesc2);

    auto startCustomBin = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, int>> customBinMatches = matchBinaryKeypoints(briskDesc1, briskDesc2);
    auto endCustomBin = std::chrono::high_resolution_clock::now();

    auto startCvBin = std::chrono::high_resolution_clock::now();
    cv::BFMatcher bfMatcherHamming(cv::NORM_HAMMING);
    std::vector<cv::DMatch> cvBinMatches;
    bfMatcherHamming.match(briskDesc1, briskDesc2, cvBinMatches);
    auto endCvBin = std::chrono::high_resolution_clock::now();

    double timeCustomBin = std::chrono::duration<double, std::milli>(endCustomBin - startCustomBin).count();
    double timeCvBin = std::chrono::duration<double, std::milli>(endCvBin - startCvBin).count();

    std::cout << "Execution Times (ms):\n"
              << "Custom Binary: " << timeCustomBin << "\n"
              << "OpenCV Binary: " << timeCvBin << "\n";
    
    std::vector<std::pair<int, int>> commonBinMatches = findCommonMatches(customBinMatches, cvBinMatches);
    std::cout << " | Custom Binary Matches: " << customBinMatches.size()
              << " | CV Binary Matches: " << cvBinMatches.size()
              << " | Common Binary Matches: " << commonBinMatches.size() << std::endl;

    std::vector<cv::DMatch> customBinDMatches = convertToDMatch(customBinMatches);

    auto ransacBinMatches = applyRANSAC(customBinDMatches, briskKps1, briskKps2);

    cv::Mat floatMatchesImg, binaryMatchesImg;;
    cv::drawMatches(img1, briskKps1, img2, briskKps2, ransacBinMatches, binaryMatchesImg);

    cv::imshow("BRISK Matches", binaryMatchesImg);
    cv::waitKey(0);

    return 0;
}