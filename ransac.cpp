#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <future>
#include <mutex>
#include <random>
#include "feature_extraction_parallel_GPU/threadpool.h"
#include "ransac.hpp"


double computeSampsonError(const Eigen::Matrix3d& F,
                           const Eigen::Vector3d& x,
                           const Eigen::Vector3d& x_prime)
{
    Eigen::Vector3d Fx = F * x;
    Eigen::Vector3d Ftx = F.transpose() * x_prime;
    double numerator = std::pow(x_prime.transpose() * F * x, 2);
    double denominator = Fx(0) * Fx(0) + Fx(1) * Fx(1) + Ftx(0) * Ftx(0) + Ftx(1) * Ftx(1);

    if (denominator < 1e-12) return std::numeric_limits<double>::max();
    return numerator / denominator;
}

void FundamentalMatrix::normalizePoints(const std::vector<std::pair<Point, Point>>& points,
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
        scale1 += std::pow(p.first.x - meanX1, 2) + std::pow(p.first.y - meanY1, 2);
        scale2 += std::pow(p.second.x - meanX2, 2) + std::pow(p.second.y - meanY2, 2);
    }
    scale1 = std::sqrt(2) / std::sqrt(scale1 / points.size());
    scale2 = std::sqrt(2) / std::sqrt(scale2 / points.size());


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

Eigen::Matrix3d FundamentalMatrix::computeFundamentalMatrix(std::vector<std::pair<Point, Point>> points) {
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

    return F;
}

void FundamentalMatrix::fit(const std::vector<std::pair<Point, Point>>& sample) {
    if (sample.size() < 8) return;
    F = computeFundamentalMatrix(sample);
    inliers = sample;
}

Eigen::Matrix3d FundamentalMatrix::getMatrix() const {
    return F;
}

int FundamentalMatrix::countInliers(const std::vector<std::pair<Point, Point>>& data, double threshold) const {
    int inliers = 0;
    for (const auto& pair : data) {
        Eigen::Vector3d p1(pair.first.x, pair.first.y, 1.0);
        Eigen::Vector3d p2(pair.second.x, pair.second.y, 1.0);
        double error = std::abs(p2.transpose() * F * p1);
        inliers += error < threshold;
    }
    return inliers;
}

const std::vector<std::pair<Point, Point>>& FundamentalMatrix::getInliers() const {
    return inliers;
}

void Ransac::run(FundamentalMatrix& model,
                 const std::vector<std::pair<Point, Point>>& data,
                 double probability,
                 double sampsonThreshold,
                 int numThreads,
                 thread_pool& pool) {

    int actualIterations = 0;
    int N = data.size();
    int sampleSize = 8;
    double outlierRatio = 0.5;
    int maxIterations = std::log(1.0 - probability) / std::log(1.0 - std::pow(1.0 - outlierRatio, sampleSize));

    int bestInliers = 0;
    std::vector<std::pair<Point, Point>> bestInlierSet;
    Eigen::Matrix3d bestF;

    std::mt19937 rng(std::random_device{}());

    for (int iter = 0; iter < maxIterations; ++iter) {
        ++actualIterations;
        std::vector<std::pair<Point, Point>> sample;
        std::sample(data.begin(), data.end(), std::back_inserter(sample), sampleSize, rng);

        FundamentalMatrix tempModel;
        tempModel.fit(sample);

        Eigen::Matrix3d F = tempModel.getMatrix();

        std::mutex inlierMutex;
        std::vector<std::pair<Point, Point>> currentInliers;

        int chunkSize = data.size() / numThreads;
        std::vector<std::future<void>> futures;

        for (int t = 0; t < numThreads; ++t) {
            int start = t * chunkSize;
            int end = std::min(static_cast<int>(data.size()), (t + 1) * chunkSize);

            futures.emplace_back(pool.submit([&, start, end]() {
                std::vector<std::pair<Point, Point>> localInliers;
                for (int i = start; i < end; ++i) {
                    const auto& pair = data[i];
                    Eigen::Vector3d x(pair.first.x, pair.first.y, 1.0);
                    Eigen::Vector3d x_prime(pair.second.x, pair.second.y, 1.0);

                    double error = computeSampsonError(F, x, x_prime);
                    if (error < sampsonThreshold) {
                        localInliers.emplace_back(pair);
                    }
                }

                std::lock_guard<std::mutex> lock(inlierMutex);
                currentInliers.insert(currentInliers.end(),
                                      localInliers.begin(), localInliers.end());
            }));
        }
        for (auto& f : futures) f.get();

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

    model.fit(bestInlierSet);
}
