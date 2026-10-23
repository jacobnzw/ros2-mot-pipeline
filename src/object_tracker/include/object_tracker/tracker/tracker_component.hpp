#pragma once

#include "object_tracker/msg/tracked_object_array.hpp"
#include "object_tracker/tracker/byte_tracker.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <cstdint>
#include <mutex>

namespace object_tracker {

class TrackerComponent : public rclcpp::Node {
public:
  explicit TrackerComponent(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  struct Latencies {
    int64_t detections_stamp_ns;
    double callback_interval_ms = 0.0;
    int64_t frame_drop_count = 0;
    bool had_previous_detections = false;
  };

  void detectionsCallback(const vision_msgs::msg::Detection2DArray::UniquePtr msg);
  Latencies measure_latencies(const rclcpp::Time &detections_stamp);
  std::vector<Detection> convert(const vision_msgs::msg::Detection2DArray &detections_msg) const;
  msg::TrackedObjectArray convert(const std::vector<TrackedObject> &tracked_objects,
                                  const std_msgs::msg::Header &header) const;

  ByteTracker tracker_;

  rclcpp::Clock::SharedPtr metrics_clock_;
  std::mutex metrics_mutex_;

  double expected_frame_rate_{10.0};
  bool has_previous_detections_{false};
  int64_t previous_detections_stamp_ns_{0};
  int64_t previous_callback_start_ns_{0};
  int64_t frame_drop_count_{0};

  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Publisher<object_tracker::msg::TrackedObjectArray>::SharedPtr tracks_pub_;
};

} // namespace object_tracker
