#pragma once

#include "object_tracker/msg/tracked_object_array.hpp"
#include "object_tracker/tracker/byte_tracker.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace object_tracker {

class TrackerComponent : public rclcpp::Node {
public:
  explicit TrackerComponent(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  void detectionsCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr msg);
  std::vector<Detection> convert(const vision_msgs::msg::Detection2DArray &detections_msg) const;
  msg::TrackedObjectArray convert(const std::vector<TrackedObject> &tracked_objects,
                                  const std_msgs::msg::Header &header) const;

  ByteTracker tracker_;

  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Publisher<object_tracker::msg::TrackedObjectArray>::SharedPtr tracks_pub_;
};

} // namespace object_tracker
