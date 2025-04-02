#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

struct Point {
    double x, y;
};

class FundamentalMatrix {
private:
    Eigen::Matrix3d F;

    static Eigen::Matrix3d computeFundamentalMatrix(std::vector<std::pair<Point, Point>> points) {
        Eigen::Matrix3d T1, T2;
        std::vector<Eigen::Vector3d> normPoints1, normPoints2;
        normalizePoints(points, normPoints1, normPoints2, T1, T2);

        Eigen::MatrixXd A(points.size(), 9);
        for (size_t i = 0; i < points.size(); ++i) {
            Eigen::Vector3d p1 = normPoints1[i];
            Eigen::Vector3d p2 = normPoints2[i];
            A.row(i) << p1.x() * p2.x(), p1.x() * p2.y(), p1.x(),
                        p1.y() * p2.x(), p1.y() * p2.y(), p1.y(),
                        p2.x(), p2.y(), 1;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd f = svd.matrixV().col(8);

        Eigen::Matrix3d F;
        F << f(0), f(1), f(2),
             f(3), f(4), f(5),
             f(6), f(7), f(8);

        F = T2.transpose() * F * T1;

        Eigen::JacobiSVD<Eigen::Matrix3d> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d singularValues = svdF.singularValues();
        singularValues(2) = 0;
        F = svdF.matrixU() * singularValues.asDiagonal() * svdF.matrixV().transpose();

        Eigen::JacobiSVD<Eigen::Matrix3d> checkSVD(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        std::cout << "Singular values of final F: " << checkSVD.singularValues().transpose() << std::endl;

        return F;
    }

    static void normalizePoints(const std::vector<std::pair<Point, Point>>& points,
                                std::vector<Eigen::Vector3d>& normPoints1,
                                std::vector<Eigen::Vector3d>& normPoints2,
                                Eigen::Matrix3d& T1, Eigen::Matrix3d& T2) {
        double meanX1 = 0, meanY1 = 0, meanX2 = 0, meanY2 = 0;
        for (const auto& p : points) {
            meanX1 += p.first.x;
            meanY1 += p.first.y;
            meanX2 += p.second.x;
            meanY2 += p.second.y;
        }
        meanX1 /= points.size();
        meanY1 /= points.size();
        meanX2 /= points.size();
        meanY2 /= points.size();

        double scale1 = 0, scale2 = 0;
        for (const auto& p : points) {
            scale1 += std::sqrt(std::pow(p.first.x - meanX1, 2) + std::pow(p.first.y - meanY1, 2));
            scale2 += std::sqrt(std::pow(p.second.x - meanX2, 2) + std::pow(p.second.y - meanY2, 2));
        }
        scale1 = std::sqrt(2) / (scale1 / points.size());
        scale2 = std::sqrt(2) / (scale2 / points.size());

        T1 << scale1, 0, -scale1 * meanX1,
              0, scale1, -scale1 * meanY1,
              0, 0, 1;
        T2 << scale2, 0, -scale2 * meanX2,
              0, scale2, -scale2 * meanY2,
              0, 0, 1;

        for (const auto& p : points) {
            normPoints1.push_back(T1 * Eigen::Vector3d(p.first.x, p.first.y, 1.0));
            normPoints2.push_back(T2 * Eigen::Vector3d(p.second.x, p.second.y, 1.0));
        }
    }

public:
    void fit(const std::vector<std::pair<Point, Point>>& sample) {
        if (sample.size() < 8) return;
        F = computeFundamentalMatrix(sample);
    }

    Eigen::Matrix3d getMatrix() const {
        return F;
    }

    int countInliers(const std::vector<std::pair<Point, Point>>& data, double threshold) const {
        int inliers = 0;
        for (const auto& pair : data) {
            Eigen::Vector3d p1(pair.first.x, pair.first.y, 1.0);
            Eigen::Vector3d p2(pair.second.x, pair.second.y, 1.0);
            double error = std::abs(p2.transpose() * F * p1);
            inliers += error < threshold;
        }
        return inliers;
    }

    void print() const {
        std::cout << "Fundamental Matrix:\n" << F << std::endl;
    }
};

double computeSampsonError(const Eigen::Matrix3d& F,
                           const Eigen::Vector3d& x,
                           const Eigen::Vector3d& x_prime)
{
    Eigen::Vector3d Fx = F * x;
    Eigen::Vector3d Ftx = F.transpose() * x_prime;
    double numerator = std::pow(x_prime.transpose() * F * x, 2);
    double denominator = Fx(0) * Fx(0) + Fx(1) * Fx(1) + Ftx(0) * Ftx(0) + Ftx(1) * Ftx(1);

    if (denominator < 1e-12) return std::numeric_limits<double>::max(); // Avoid division by zero
    return numerator / denominator;
}

class Ransac {
public:
    static void run(FundamentalMatrix& model,
                    const std::vector<std::pair<Point, Point>>& data,
                    double probability,
                    double sampsonThreshold) {
        int N = data.size();
        int sampleSize = 8;
        double outlierRatio = 0.5;
        int maxIterations = std::log(1.0 - probability) / std::log(1.0 - std::pow(1.0 - outlierRatio, sampleSize));

        int bestInliers = 0;
        std::vector<std::pair<Point, Point>> bestInlierSet;
        Eigen::Matrix3d bestF;

        std::mt19937 rng(std::random_device{}());

        for (int iter = 0; iter < maxIterations; ++iter) {
            std::vector<std::pair<Point, Point>> sample;
            std::sample(data.begin(), data.end(), std::back_inserter(sample),
                        sampleSize, rng);

            FundamentalMatrix tempModel;
            tempModel.fit(sample);
            Eigen::Matrix3d F = tempModel.getMatrix();

            std::vector<std::pair<Point, Point>> currentInliers;
            for (const auto& pair : data) {
                Eigen::Vector3d x(pair.first.x, pair.first.y, 1.0);
                Eigen::Vector3d x_prime(pair.second.x, pair.second.y, 1.0);

                double error = computeSampsonError(F, x, x_prime);
                if (error < sampsonThreshold) {
                    currentInliers.push_back(pair);
                }
            }

            if (currentInliers.size() > bestInliers) {
                bestInliers = currentInliers.size();
                bestInlierSet = currentInliers;
                bestF = F;

                outlierRatio = 1.0 - static_cast<double>(bestInliers) / N;
                double denom = std::log(1.0 - std::pow(1.0 - outlierRatio, sampleSize));
                if (denom != 0.0) {
                    maxIterations = std::log(1.0 - probability) / denom;
                    maxIterations = std::clamp(maxIterations, 100, 2000);
                }
            }
        }

        std::cout << "Best Sampson inliers: " << bestInliers << " / " << N << std::endl;

        model.fit(bestInlierSet);
        model.print();
    }
};

//for testing

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
    std::vector<std::pair<int, int>> pointMatches;
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
            pointMatches.emplace_back(i, bestIdx);
        }
    }
    return pointMatches;
}

int checkRank(const Eigen::MatrixXd& matrix) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd singularValues = svd.singularValues();
    int rank = (singularValues.array() > 1e-10).count();
    std::cout << "Singular values: " << singularValues.transpose() << std::endl;
    std::cout << "Rank of the matrix: " << rank << std::endl;

    return rank;
}

int main() {
    cv::Mat img1 = cv::imread("/Users/ostappavlyshyn/CLionProjects/ransac/first.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("/Users/ostappavlyshyn/CLionProjects/ransac/second.png", cv::IMREAD_GRAYSCALE);

    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();

    std::vector<cv::KeyPoint> briskKps1, briskKps2;
    cv::Mat briskDesc1, briskDesc2;

    brisk->detectAndCompute(img1, cv::noArray(), briskKps1, briskDesc1);
    brisk->detectAndCompute(img2, cv::noArray(), briskKps2, briskDesc2);

    auto matches = matchBinaryKeypoints(briskDesc1, briskDesc2);

    std::vector<std::pair<Point, Point>> matchedPoints;
    for (const auto& match : matches) {
        Point p1 = {briskKps1[match.first].pt.x, briskKps1[match.first].pt.y};
        Point p2 = {briskKps2[match.second].pt.x, briskKps2[match.second].pt.y};

        matchedPoints.push_back({p1, p2});
    }

    FundamentalMatrix fundamentalModel;
    Ransac::run(fundamentalModel, matchedPoints, 0.99, 1.0);

    int sampsonInliers = 0;
    int sampsonOutliers = 0;
    double sampsonThreshold = 1.0;

    Eigen::Matrix3d F = fundamentalModel.getMatrix();

    for (const auto& pair : matchedPoints) {
        Eigen::Vector3d x(pair.first.x, pair.first.y, 1.0);
        Eigen::Vector3d x_prime(pair.second.x, pair.second.y, 1.0);

        double error = computeSampsonError(F, x, x_prime);
        if (error < sampsonThreshold) {
            ++sampsonInliers;
        } else {
            ++sampsonOutliers;
        }
    }

    std::cout << "Sampson error inliers: " << sampsonInliers << std::endl;
    std::cout << "Sampson error outliers: " << sampsonOutliers << std::endl;

    int rank = checkRank(fundamentalModel.getMatrix());
    std::cout << "Rank of F: " << rank << std::endl;

    return 0;
}
