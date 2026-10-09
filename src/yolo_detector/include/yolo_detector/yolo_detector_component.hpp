#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <memory>

#include "yolo_detector/onnx_model.hpp"

namespace yolo_detector {

class YoloDetectorComponent : public rclcpp::Node {
public:
  explicit YoloDetectorComponent(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);

  std::shared_ptr<OnnxModel> model_;
  double confidence_threshold_{0.25};
  double nms_threshold_{0.45};
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;
};

} // namespace yolo_detector
