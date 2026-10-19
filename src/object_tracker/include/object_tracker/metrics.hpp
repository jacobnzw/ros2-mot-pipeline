#pragma once

#include <rclcpp/rclcpp.hpp>

#include <string_view>

namespace object_tracker::metrics {

void initialize_rerun();
void log_scalar(std::string_view entity_path, double value);
rclcpp::Clock::SharedPtr create_clock(rclcpp::Node &node);

} // namespace object_tracker::metrics
