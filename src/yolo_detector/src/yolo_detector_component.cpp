#include "yolo_detector/yolo_detector_component.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <fmt/ranges.h>
#include <sstream>
#include <stdexcept>

namespace yolo_detector {

YoloDetectorComponent::YoloDetectorComponent(const rclcpp::NodeOptions &options)
    : Node("yolo_detector_component", options) {

  // Input parameters modifiable during: ros2 component load
  // Example: -p model_path:=model/yolo/yolo11n.onnx -p image_topic:=/camera/image_raw
  declare_parameter("model_path", std::string(""));
  declare_parameter("image_topic", std::string("/image_raw"));
  declare_parameter("detection_topic", std::string("~/detections"));

  const auto model_path = get_parameter("model_path").as_string();
  if (model_path.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter 'model_path' is required before "
                               "loading the detector component.");
    return;
  }

  // Allocate an OnnxModel on the heap, constructs it with the path, and wrap w/ shared_ptr.
  try {
    model_ = std::make_shared<OnnxModel>(model_path);
  } catch (const std::exception &ex) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize the ONNX model from '%s': %s",
                 model_path.c_str(), ex.what());
    throw;
  }

  // Create subscription for the image_topic and bind callback
  const auto image_topic = get_parameter("image_topic").as_string();
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS(),
      std::bind(&YoloDetectorComponent::imageCallback, this, std::placeholders::_1));

  // Create publisher for the detection_topic
  const auto detection_topic = get_parameter("detection_topic").as_string();
  detections_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(detection_topic, 10);

  RCLCPP_INFO(get_logger(), "YOLO detector component initialized with model '%s'.",
              model_path.c_str());
}

void YoloDetectorComponent::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  if (!model_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Detector model is not loaded.");
    return;
  }

  try {
    // Model inference: run the incomming image through the YOLO detector
    const auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    const auto output_shapes = model_->infer(cv_ptr->image);

    // Output empty detections for now
    // TODO: Fill the detections output w/ decoded YOLO detections
    vision_msgs::msg::Detection2DArray detections;
    detections.header = msg->header;
    detections.detections.clear();

    const auto shapes_text = fmt::format("{}", output_shapes);
    RCLCPP_INFO(get_logger(), "Processed image with output tensor shapes: %s", shapes_text.c_str());

    detections_pub_->publish(detections);

  } catch (const std::exception &ex) {
    RCLCPP_ERROR(get_logger(), "Inference failed: %s", ex.what());
  }
}

} // namespace yolo_detector

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_detector::YoloDetectorComponent)
