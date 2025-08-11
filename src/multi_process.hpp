
#ifndef MULTI_PREPROCESS_HPP
#define MULTI_PREPROCESS_HPP
#include <Eigen/Dense>
#include <vector>
#include <cmath>

struct RigidResult 
{
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  double rms = 0.0;
  bool ok = false;
};

inline RigidResult SolveRigidTransformWeighted(
    const std::vector<Eigen::Vector3d>& lidar_pts,
    const std::vector<Eigen::Vector3d>& cam_pts,
    const std::vector<double>* weights = nullptr)
{
    RigidResult out; out.ok = false;
    const size_t N = lidar_pts.size();
    if (N < 3 || cam_pts.size() != N) return out;

    std::vector<double> w(N, 1.0);
    if (weights && weights->size() == N) w = *weights;
    double wsum = 0.0;
    for (double wi : w) wsum += wi;
    if (wsum <= 0) return out;

    Eigen::Vector3d muL = Eigen::Vector3d::Zero();
    Eigen::Vector3d muC = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < N; ++i) 
    {
        muL += w[i] * lidar_pts[i];
        muC += w[i] * cam_pts[i];
    }
    muL /= wsum; muC /= wsum;

    Eigen::Matrix3d Sigma = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < N; ++i) 
    {
        Eigen::Vector3d l = lidar_pts[i] - muL;
        Eigen::Vector3d c = cam_pts[i] - muC;
        Sigma += w[i] * (l * c.transpose());
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0) 
    {
        Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
        D(2,2) = -1;
        R = V * D * U.transpose();
    }
    Eigen::Vector3d t = muC - R * muL;

    double rss = 0.0;
    for (size_t i = 0; i < N; ++i) 
    {
        Eigen::Vector3d r = (R * lidar_pts[i] + t) - cam_pts[i];
        rss += w[i] * r.squaredNorm();
    }
    out.R = R; out.t = t; out.rms = std::sqrt(rss / wsum); out.ok = true;
    return out;

}
#endif
