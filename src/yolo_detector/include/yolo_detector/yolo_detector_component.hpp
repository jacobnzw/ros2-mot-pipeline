#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <memory>
#include <string>

#include "yolo_detector/onnx_model.hpp"

namespace yolo_detector {

class YoloDetectorComponent : public rclcpp::Node {
public:
  explicit YoloDetectorComponent(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  std::shared_ptr<OnnxModel> model_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;
};

} // namespace yolo_detector
