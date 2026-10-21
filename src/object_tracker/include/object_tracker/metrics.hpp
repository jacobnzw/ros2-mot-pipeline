#pragma once

#include <rclcpp/logger.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <cstdint>
#include <span>
#include <string_view>

namespace object_tracker::metrics {

void initialize_rerun();
void log_scalar(std::string_view entity_path, double value, int64_t ros_timestamp_ns);
void log_encoded_image(std::string_view entity_path, const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg,
                       int64_t ros_timestamp_ns);
void log_detections(std::string_view entity_path, const vision_msgs::msg::Detection2DArray &detections,
                    std::span<const char *const> class_names, int64_t ros_timestamp_ns);
rclcpp::Clock::SharedPtr create_clock(rclcpp::Node &node);

} // namespace object_tracker::metrics
