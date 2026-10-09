#pragma once

#include <foxglove_msgs/msg/image_annotations.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace yolo_detector {

class YoloDetectorObserver : public rclcpp::Node {
public:
  explicit YoloDetectorObserver(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  void foxgloveConversionCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr msg);

  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr annotations_pub_;
};

} // namespace yolo_detector
