#pragma once

#include <rclcpp/rclcpp.hpp>

#include <cstdint>
#include <span>
#include <string_view>

namespace object_tracker::metrics {

void initialize_rerun();
void log_scalar(std::string_view entity_path, double value, int64_t ros_timestamp_ns);
void log_encoded_image(std::string_view entity_path, std::span<const uint8_t> image_data, std::string_view format,
                       int64_t ros_timestamp_ns);
rclcpp::Clock::SharedPtr create_clock(rclcpp::Node &node);

} // namespace object_tracker::metrics
