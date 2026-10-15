#pragma once

#include <Eigen/Core>

#include "object_tracker/tracker/detection.hpp"

namespace object_tracker {

// Constant-velocity Kalman filter over (centre x, centre y, aspect ratio, height) - the motion
// model used by SORT, DeepSORT and ByteTrack. Process and measurement noise scale with box height,
// so the filter trusts pixel motion equally for near (large) and far (small) objects.
class BoxKalmanFilter {
public:
  using Mean = Eigen::Matrix<float, 8, 1>; // TODO: Why 8D? we have x,y, aspect, height that's 4
  using Covariance = Eigen::Matrix<float, 8, 8>;
  using Measurement = Eigen::Matrix<float, 4, 1>;

  struct State {
    Mean mean;
    Covariance covariance;
  };

  [[nodiscard]] static State initiate(const Measurement &z) noexcept;
  static void predict(State &state) noexcept;
  // Leaves the state unchanged if the innovation covariance is not positive definite.
  static void update(State &state, const Measurement &z) noexcept;

  [[nodiscard]] static Measurement to_measurement(const Box &box) noexcept;
  [[nodiscard]] static Box to_box(const Mean &mean) noexcept;

private:
  static constexpr float kStdWeightPosition = 1.0f / 20.0f;
  static constexpr float kStdWeightVelocity = 1.0f / 160.0f;
};

} // namespace object_tracker