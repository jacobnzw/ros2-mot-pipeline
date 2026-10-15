#include "object_tracker/tracker/kalman_filter.hpp"

#include <Eigen/Cholesky>

namespace object_tracker {
namespace {

constexpr float square(float v) noexcept { return v * v; }

// x' = F x with unit time step: position += velocity.
Eigen::Matrix<float, 8, 8> motion_model() noexcept {
  Eigen::Matrix<float, 8, 8> f = Eigen::Matrix<float, 8, 8>::Identity();
  for (Eigen::Index i = 0; i < 4; ++i)
    f(i, i + 4) = 1.0f;
  return f;
}

} // namespace

BoxKalmanFilter::State BoxKalmanFilter::initiate(const Measurement &z) noexcept {
  State s;
  s.mean.setZero();
  s.mean.head<4>() = z;

  const float h = z(3, 0);
  const float pos = kStdWeightPosition * h;
  const float vel = kStdWeightVelocity * h;
  Eigen::Matrix<float, 8, 1> variances;
  variances << square(2 * pos), square(2 * pos), square(1e-2f), square(2 * pos), square(10 * vel), square(10 * vel),
      square(1e-5f), square(10 * vel);
  s.covariance = variances.asDiagonal();
  return s;
}

void BoxKalmanFilter::predict(State &s) noexcept {
  static const Eigen::Matrix<float, 8, 8> kF = motion_model();

  const float h = s.mean(3, 0);
  const float pos = kStdWeightPosition * h;
  const float vel = kStdWeightVelocity * h;
  Eigen::Matrix<float, 8, 1> variances;
  variances << square(pos), square(pos), square(1e-2f), square(pos), square(vel), square(vel), square(1e-5f),
      square(vel);
  const Covariance process_noise = variances.asDiagonal();

  s.mean = kF * s.mean;
  s.covariance = kF * s.covariance * kF.transpose() + process_noise;
}

void BoxKalmanFilter::update(State &s, const Measurement &z) noexcept {
  const float h = s.mean(3, 0);
  const float pos = kStdWeightPosition * h;
  Eigen::Matrix<float, 4, 1> variances;
  variances << square(pos), square(pos), square(1e-1f), square(pos);
  const Eigen::Matrix<float, 4, 4> measurement_noise = variances.asDiagonal();

  // The measurement matrix H just selects the first four state entries, so H P H^T and P H^T are
  // sub-blocks of P - no need to form H explicitly.
  const Eigen::Matrix<float, 8, 4> p_ht = s.covariance.leftCols<4>();
  const Eigen::Matrix<float, 4, 4> innovation_cov = s.covariance.topLeftCorner<4, 4>() + measurement_noise;

  // K = P H^T S^-1  <=>  S K^T = (P H^T)^T, since S is symmetric.
  Eigen::LLT<Eigen::Matrix<float, 4, 4>> llt(innovation_cov);
  if (llt.info() != Eigen::Success)
    return;
  const Eigen::Matrix<float, 4, 8> k_transposed = llt.solve(p_ht.transpose());
  const Eigen::Matrix<float, 8, 4> gain = k_transposed.transpose();

  const Measurement innovation = z - s.mean.head<4>();
  s.mean += gain * innovation;
  s.covariance -= gain * p_ht.transpose(); // P - K S K^T == P - K (P H^T)^T
}

BoxKalmanFilter::Measurement BoxKalmanFilter::to_measurement(const Box &box) noexcept {
  Measurement z;
  const float h = box.height();
  z(0, 0) = box.center_x();
  z(1, 0) = box.center_y();
  z(2, 0) = h > 0.0f ? box.width() / h : 0.0f;
  z(3, 0) = h;
  return z;
}

Box BoxKalmanFilter::to_box(const Mean &mean) noexcept {
  const float h = mean(3, 0);
  const float w = mean(2, 0) * h;
  return Box::from_center(mean(0, 0), mean(1, 0), w, h);
}

} // namespace object_tracker