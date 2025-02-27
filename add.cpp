#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <algorithm>

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
            A.row(i) << p1.x() * p2.x(), p1.x() * p2.y(), p1.x(), p1.y() * p2.x(), p1.y() * p2.y(), p1.y(), p2.x(), p2.y(), 1;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd f = svd.matrixV().col(8);

        Eigen::Matrix3d F;
        F << f(0), f(1), f(2),
             f(3), f(4), f(5),
             f(6), f(7), f(8);

        Eigen::JacobiSVD<Eigen::Matrix3d> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d singularValues = svdF.singularValues();
        singularValues(2) = 0;
        F = svdF.matrixU() * singularValues.asDiagonal() * svdF.matrixV().transpose();

        return T2.transpose() * F * T1;
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

    int countInliers(const std::vector<std::pair<Point, Point>>& data, double threshold) const {
        int inliers = 0;
        for (const auto& pair : data) {
            Eigen::Vector3d p1(pair.first.x, pair.first.y, 1.0);
            Eigen::Vector3d p2(pair.second.x, pair.second.y, 1.0);
            double error = std::abs(p2.transpose() * F * p1);
            if (error < threshold) {
                inliers++;
            }
        }
        return inliers;
    }

    void print() const {
        std::cout << "Fundamental Matrix:\n" << F << std::endl;
    }
};

class Ransac {
public:
    static void run(FundamentalMatrix& model, std::vector<std::pair<Point, Point>>& data, double probability, double threshold) {
        int N = data.size();
        double w = 0.5;
        int sampleSize = 8;
        int bestInliers = 0;
        std::srand(std::time(0));
        int maxIterations = std::log(1 - probability) / std::log(1 - std::pow(w, sampleSize));

        for (int i = 0; i < maxIterations; ++i) {
            std::vector<std::pair<Point, Point>> sample;
            std::sample(data.begin(), data.end(), std::back_inserter(sample), sampleSize, std::mt19937{std::random_device{}()});

            model.fit(sample);
            int inliers = model.countInliers(data, threshold);

            if (inliers > bestInliers) {
                bestInliers = inliers;
                w = static_cast<double>(inliers) / N;
                maxIterations = std::log(1 - probability) / std::log(1 - std::pow(w, sampleSize));
            }
        }

        std::cout << "Best Model with " << bestInliers << " inliers:\n";
        model.print();
    }
};

int main() {
    std::vector<std::pair<Point, Point>> data = {{{1, 2}, {2, 3}}, {{2, 3}, {3, 4}}, {{3, 4}, {4, 5}},
                                                 {{4, 5}, {5, 6}}, {{5, 6}, {6, 7}}, {{10, 15}, {15, 20}},
                                                 {{15, 20}, {20, 25}}, {{30, 40}, {35, 45}}};

    FundamentalMatrix fundamentalModel;
    Ransac::run(fundamentalModel, data, 0.99, 1.0);

    return 0;
}