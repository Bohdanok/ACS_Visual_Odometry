#ifndef ransac_hpp
#define ransac_hpp

#include <Eigen/Dense>
#include <vector>
#include <utility>

struct Point {
    double x, y;
};

class FundamentalMatrix {
private:
    Eigen::Matrix3d F;
    std::vector<std::pair<Point, Point>> inliers;

    static Eigen::Matrix3d computeFundamentalMatrix(std::vector<std::pair<Point, Point>> points);
    static void normalizePoints(const std::vector<std::pair<Point, Point>>& points,
                                std::vector<Eigen::Vector3d>& normPoints1,
                                std::vector<Eigen::Vector3d>& normPoints2,
                                Eigen::Matrix3d& T1, Eigen::Matrix3d& T2);

public:
    void fit(const std::vector<std::pair<Point, Point>>& sample);
    Eigen::Matrix3d getMatrix() const;
    int countInliers(const std::vector<std::pair<Point, Point>>& data, double threshold) const;
    const std::vector<std::pair<Point, Point>>& getInliers() const;
};

double computeSampsonError(const Eigen::Matrix3d& F,
                           const Eigen::Vector3d& x,
                           const Eigen::Vector3d& x_prime);

class Ransac {
public:
    void run(FundamentalMatrix& model,
             const std::vector<std::pair<Point, Point>>& data,
             double probability,
             double sampsonThreshold);
};

#endif
