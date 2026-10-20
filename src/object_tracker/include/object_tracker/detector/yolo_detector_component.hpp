#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "object_tracker/detector/onnx_model.hpp"

#include <cstdint>
#include <mutex>

namespace object_tracker {

class YoloDetectorComponent : public rclcpp::Node {
public:
  explicit YoloDetectorComponent(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);

  // TODO: WHY shared_ptr here?
  std::shared_ptr<OnnxModel> model_;

  double confidence_threshold_{0.25};
  double nms_threshold_{0.45};
  double expected_frame_rate_{10.0};
  bool log_input_image_{false};

  rclcpp::Clock::SharedPtr metrics_clock_;
  std::mutex metrics_mutex_;

  bool has_previous_image_{false};
  int64_t previous_image_stamp_ns_{0};
  int64_t previous_callback_start_ns_{0};
  int64_t frame_drop_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;
};

} // namespace object_tracker
