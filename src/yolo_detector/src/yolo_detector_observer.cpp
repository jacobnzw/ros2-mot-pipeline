
#include <rclcpp/rclcpp.hpp>

#include "yolo_detector/yolo_detector_observer.hpp"
#include <rclcpp_components/register_node_macro.hpp>

#include <cmath>
#include <cstddef>
#include <utility>

namespace yolo_detector {
namespace {

foxglove_msgs::msg::Point2 make_bbox_point(double center_x, double center_y, double offset_x, double offset_y,
                                           double theta) {
  foxglove_msgs::msg::Point2 point;
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  point.x = center_x + offset_x * cos_theta - offset_y * sin_theta;
  point.y = center_y + offset_x * sin_theta + offset_y * cos_theta;
  return point;
}

} // namespace

YoloDetectorObserver::YoloDetectorObserver(const rclcpp::NodeOptions &options)
    : Node("yolo_detector_observer", options) {

  declare_parameter("detections_topic", "/detections");
  declare_parameter("foxglove_topic", "/detection_annotations");

  // Create subscription for the image_topic and bind callback
  const auto detections_topic = get_parameter("detections_topic").as_string();
  detections_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
      detections_topic, rclcpp::SensorDataQoS(),
      std::bind(&YoloDetectorObserver::foxgloveConversionCallback, this, std::placeholders::_1));

  // Create publisher for the detection_topic
  const auto annotation_topic = get_parameter("foxglove_topic").as_string();
  annotations_pub_ = create_publisher<foxglove_msgs::msg::ImageAnnotations>(annotation_topic, 10);
}

void YoloDetectorObserver::foxgloveConversionCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) {
  foxglove_msgs::msg::ImageAnnotations annotations;
  annotations.timestamp = msg->header.stamp;

  for (std::size_t index = 0; index < msg->detections.size(); ++index) {
    const auto &bbox = msg->detections[index].bbox;
    const double center_x = bbox.center.position.x;
    const double center_y = bbox.center.position.y;
    const double theta = bbox.center.theta;
    const double half_width = bbox.size_x * 0.5;
    const double half_height = bbox.size_y * 0.5;
    if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(theta) || !std::isfinite(bbox.size_x) ||
        !std::isfinite(bbox.size_y) || bbox.size_x <= 0.0 || bbox.size_y <= 0.0) {
      RCLCPP_WARN(get_logger(), "Skipping detection %zu with an invalid bounding box", index);
      continue;
    }

    foxglove_msgs::msg::PointsAnnotation bbox_annotation;
    bbox_annotation.type = foxglove_msgs::msg::PointsAnnotation::LINE_LOOP;
    bbox_annotation.outline_color.r = 0.1;
    bbox_annotation.outline_color.g = 1.0;
    bbox_annotation.outline_color.b = 0.2;
    bbox_annotation.outline_color.a = 1.0;
    bbox_annotation.fill_color.a = 0.0;
    bbox_annotation.thickness = 2.0;
    bbox_annotation.points = {
        make_bbox_point(center_x, center_y, -half_width, -half_height, theta),
        make_bbox_point(center_x, center_y, half_width, -half_height, theta),
        make_bbox_point(center_x, center_y, half_width, half_height, theta),
        make_bbox_point(center_x, center_y, -half_width, half_height, theta),
    };
    annotations.points.push_back(std::move(bbox_annotation));
  }

  annotations_pub_->publish(annotations);
}
} // namespace yolo_detector

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_detector::YoloDetectorObserver)
