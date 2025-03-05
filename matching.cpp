#include <iostream>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <bitset>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

const int FLOAT_DESCRIPTOR_SIZE = 32;
const int BINARY_DESCRIPTOR_SIZE = 32;
const double MATCH_THRESHOLD = 0.5;

struct FloatKeypoint {
    int x, y;
    std::vector<float> descriptor;
};

struct BinaryKeypoint {
    int x, y;
    std::vector<uint8_t> descriptor;
};

float euclideanDistance(const std::vector<float>& d1, const std::vector<float>& d2) {
    float sum = 0.0f;
    for (size_t i = 0; i < d1.size(); ++i) {
        float diff = d1[i] - d2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

int hammingDistance(const std::vector<uint8_t>& d1, const std::vector<uint8_t>& d2) {
    int distance = 0;
    for (size_t i = 0; i < d1.size(); ++i) {
        distance += __builtin_popcount(d1[i] ^ d2[i]);
    }
    return distance;
}

std::vector<std::pair<int, int> > matchBinaryKeypoints(
    const std::vector<BinaryKeypoint>& keypoints1,
    const std::vector<BinaryKeypoint>& keypoints2,
    float ratioThreshold = 0.75)
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

std::vector<std::pair<int, int>> matchFloatKeypoints(
    std::vector<FloatKeypoint>& keypoints1,
    std::vector<FloatKeypoint>& keypoints2,
    float ratioThreshold = 0.75f)
{
    std::vector<std::pair<int, int>> matches;
    for (size_t i = 0; i < keypoints1.size(); ++i) {
        int bestIdx = -1, secondBestIdx = -1;
        float bestDist = std::numeric_limits<float>::max();
        float secondBestDist = std::numeric_limits<float>::max();
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
        if (bestIdx != -1 && secondBestIdx != -1 && bestDist < ratioThreshold * secondBestDist)
            matches.emplace_back(i, bestIdx);
    }
    return matches;
}

std::vector<std::pair<int, int>> matchBinaryKeypoints(
    std::vector<BinaryKeypoint>& keypoints1,
    std::vector<BinaryKeypoint>& keypoints2,
    float ratioThreshold = 0.75f)
{
    std::vector<std::pair<int, int>> matches;
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
        if (bestIdx != -1 && secondBestIdx != -1 && bestDist < ratioThreshold * secondBestDist)
            matches.emplace_back(i, bestIdx);
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
        std::cout << "Usage: ./compare <image1> <image2>" << std::endl;
        return -1;
    }
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    
    std::vector<cv::KeyPoint> siftKps1, siftKps2, briskKps1, briskKps2;
    cv::Mat siftDesc1, siftDesc2, briskDesc1, briskDesc2;

    sift->detectAndCompute(img1, cv::Mat(), siftKps1, siftDesc1);
    sift->detectAndCompute(img2, cv::Mat(), siftKps2, siftDesc2);

    brisk->detectAndCompute(img1, cv::Mat(), briskKps1, briskDesc1);
    brisk->detectAndCompute(img2, cv::Mat(), briskKps2, briskDesc2);

    std::vector<FloatKeypoint> floatKeypoints1, floatKeypoints2;
    for (int i = 0; i < siftKps1.size(); i++)
        floatKeypoints1.push_back({(int)siftKps1[i].pt.x, (int)siftKps1[i].pt.y, std::vector<float>(siftDesc1.ptr<float>(i), siftDesc1.ptr<float>(i) + FLOAT_DESCRIPTOR_SIZE)});
    for (int i = 0; i < siftKps2.size(); i++)
        floatKeypoints2.push_back({(int)siftKps2[i].pt.x, (int)siftKps2[i].pt.y, std::vector<float>(siftDesc2.ptr<float>(i), siftDesc2.ptr<float>(i) + FLOAT_DESCRIPTOR_SIZE)});

    std::vector<std::pair<int, int>> customFloatMatches;
    auto startCustomFloat = std::chrono::high_resolution_clock::now();
    if(floatKeypoints1.size() > floatKeypoints2.size()){
        customFloatMatches = matchFloatKeypoints(floatKeypoints1, floatKeypoints2);
    }
    else {
        customFloatMatches = matchFloatKeypoints(floatKeypoints2, floatKeypoints1);
    }
    auto endCustomFloat = std::chrono::high_resolution_clock::now();
    double timeCustomFloat = std::chrono::duration<double, std::milli>(endCustomFloat - startCustomFloat).count();

    cv::BFMatcher bfMatcherL2(cv::NORM_L2);
    std::vector<cv::DMatch> cvFloatMatches;
    auto startCvFloat = std::chrono::high_resolution_clock::now();
    bfMatcherL2.match(siftDesc1, siftDesc2, cvFloatMatches);
    std::vector<std::pair<int, int>> commonFloatMatches = findCommonMatches(customFloatMatches, cvFloatMatches);
    auto endCvFloat = std::chrono::high_resolution_clock::now();
    double timeCvFloat = std::chrono::duration<double, std::milli>(endCvFloat - startCvFloat).count();

    std::vector<BinaryKeypoint> binaryKeypoints1, binaryKeypoints2;
    for (int i = 0; i < briskKps1.size(); i++)
        binaryKeypoints1.push_back({(int)briskKps1[i].pt.x, (int)briskKps1[i].pt.y, std::vector<uint8_t>(briskDesc1.ptr<uint8_t>(i), briskDesc1.ptr<uint8_t>(i) + BINARY_DESCRIPTOR_SIZE)});
    for (int i = 0; i < briskKps2.size(); i++)
        binaryKeypoints2.push_back({(int)briskKps2[i].pt.x, (int)briskKps2[i].pt.y, std::vector<uint8_t>(briskDesc2.ptr<uint8_t>(i), briskDesc2.ptr<uint8_t>(i) + BINARY_DESCRIPTOR_SIZE)});

    std::vector<std::pair<int, int>> customBinMatches;
    auto startCustomBin = std::chrono::high_resolution_clock::now();
    if(binaryKeypoints1.size() > binaryKeypoints2.size()){
        customBinMatches = matchBinaryKeypoints(binaryKeypoints1, binaryKeypoints2);
    }
    else {
        customBinMatches = matchBinaryKeypoints(binaryKeypoints2, binaryKeypoints1);
    }
    auto endCustomBin = std::chrono::high_resolution_clock::now();
    double timeCustomBin = std::chrono::duration<double, std::milli>(endCustomBin - startCustomBin).count();

    cv::BFMatcher bfMatcherHamming(cv::NORM_HAMMING);
    std::vector<cv::DMatch> cvBinMatches;
    auto startCvBin = std::chrono::high_resolution_clock::now();
    bfMatcherHamming.match(briskDesc1, briskDesc2, cvBinMatches);
    std::vector<std::pair<int, int>> commonBinMatches = findCommonMatches(customBinMatches, cvBinMatches);
    auto endCvBin = std::chrono::high_resolution_clock::now();
    double timeCvBin = std::chrono::duration<double, std::milli>(endCvBin - startCvBin).count();

    std::cout << "Custom Float Matches: " << customFloatMatches.size()
              << " | CV Float Matches: " << cvFloatMatches.size()
              << " | Common Float Matches: " << commonFloatMatches.size()
              << " | Custom Binary Matches: " << customBinMatches.size()
              << " | CV Binary Matches: " << cvBinMatches.size()
              << " | Common Binary Matches: " << commonBinMatches.size() << std::endl;

    if (customBinMatches.size() / std::min(static_cast<float>(binaryKeypoints1.size()), static_cast<float>(binaryKeypoints2.size())) >= MATCH_THRESHOLD) {
        std::cout << "Potentially one object detected using binary descriptors!" << std::endl;
    } else {
        std::cout << "Not enough matches to confirm a single object using binary descriptors." << std::endl;
    }

    if (customFloatMatches.size() / std::min(static_cast<float>(floatKeypoints1.size()), static_cast<float>(floatKeypoints2.size())) >= MATCH_THRESHOLD) {
        std::cout << "Potentially one object detected using float-based descriptors!" << std::endl;
    } else {
        std::cout << "Not enough matches to confirm a single object using float-based descriptors." << std::endl;
    }

    std::cout << "Execution Time (ms):\n"
              << "Custom Float Matching: " << timeCustomFloat << " ms\n"
              << "OpenCV Float Matching: " << timeCvFloat << " ms\n"
              << "Custom Binary Matching: " << timeCustomBin << " ms\n"
              << "OpenCV Binary Matching: " << timeCvBin << " ms\n";


    std::vector<cv::DMatch> customFloatDMatches = convertToDMatch(commonFloatMatches);
    std::vector<cv::DMatch> customBinDMatches = convertToDMatch(commonBinMatches);

    std::vector<cv::DMatch> ransacFloatMatches = applyRANSAC(customFloatDMatches, siftKps1, siftKps2);
    std::vector<cv::DMatch> ransacBinMatches = applyRANSAC(customBinDMatches, briskKps1, briskKps2);

    cv::Mat customFloatImgMatches, customBinaryImgMatches;
    cv::drawMatches(img1, siftKps1, img2, siftKps2, ransacFloatMatches, customFloatImgMatches);
    cv::drawMatches(img1, briskKps1, img2, briskKps2, ransacBinMatches, customBinaryImgMatches);

    cv::imshow("Custom Float Keypoint Matches", customFloatImgMatches);
    cv::imshow("Custom Binary Keypoint Matches", customBinaryImgMatches);
    cv::waitKey(0);


    return 0;
}
