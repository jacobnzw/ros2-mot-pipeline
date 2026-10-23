#include "object_tracker/metrics.hpp"

#include <fmt/format.h>
#include <rclcpp/node.hpp>
#include <rerun.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace object_tracker::metrics {
namespace { // makes recording_stream local to ::metrics namespace

rerun::RecordingStream &recording_stream() {
  // Function-local static ensures singleton-like behavior.
  // First call instantiates, subsequent calls only return.
  // In C++11 and later, that first initialization is thread-safe.
  static rerun::RecordingStream stream("ros2-mot-pipeline");
  return stream;
}

} // namespace

// TODO: make the out file path configurable
void initialize_rerun() {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    auto &stream = recording_stream();
    const auto sink_error = stream.set_sinks(rerun::FileSink{"object_tracker.rrd"});
    if (sink_error.is_err()) {
      throw std::runtime_error("Failed to configure Rerun output: " + sink_error.description);
    }
    stream.set_global();
  });
}

void log_scalar(std::string_view entity_path, double value, int64_t ros_timestamp_ns) {
  auto &stream = rerun::RecordingStream::current();
  stream.set_time_timestamp_nanos_since_epoch("ros2_timestamp", ros_timestamp_ns);
  stream.log(entity_path, rerun::Scalars(value));
}

void log_encoded_image(std::string_view entity_path, const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg,
                       int64_t ros_timestamp_ns) {
  auto &stream = rerun::RecordingStream::current();
  stream.set_time_timestamp_nanos_since_epoch("ros2_timestamp", ros_timestamp_ns);

  const auto &image_data = msg->data;
  if (image_data.empty()) {
    stream.log(entity_path, rerun::Clear::FLAT);
    return;
  }

  std::string normalized_format(msg->format);
  std::transform(normalized_format.begin(), normalized_format.end(), normalized_format.begin(),
                 [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

  std::optional<rerun::MediaType> media_type;
  if (normalized_format.find("jpeg") != std::string::npos || normalized_format.find("jpg") != std::string::npos) {
    media_type = rerun::MediaType::jpeg();
  } else if (normalized_format.find("png") != std::string::npos) {
    media_type = rerun::MediaType::png();
  }

  const auto encoded_image = rerun::EncodedImage::from_bytes(
      rerun::Collection<uint8_t>::borrow(image_data.data(), image_data.size()), media_type);
  stream.log(entity_path, encoded_image);
}

void log_detections(std::string_view entity_path, const vision_msgs::msg::Detection2DArray &detections,
                    std::span<const char *const> class_names, int64_t ros_timestamp_ns) {
  auto &stream = rerun::RecordingStream::current();
  stream.set_time_timestamp_nanos_since_epoch("ros2_timestamp", ros_timestamp_ns);

  if (detections.detections.empty()) {
    stream.log(entity_path, rerun::Clear::FLAT);
    return;
  }

  std::vector<rerun::Position2D> centers;
  std::vector<rerun::HalfSize2D> half_sizes;
  std::vector<rerun::Text> labels;
  centers.reserve(detections.detections.size());
  half_sizes.reserve(detections.detections.size());
  labels.reserve(detections.detections.size());

  for (const auto &detection : detections.detections) {
    const auto &center = detection.bbox.center.position;
    const double width = detection.bbox.size_x;
    const double height = detection.bbox.size_y;
    if (!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(width) || !std::isfinite(height) ||
        width <= 0.0 || height <= 0.0) {
      continue;
    }

    centers.emplace_back(static_cast<float>(center.x), static_cast<float>(center.y));
    half_sizes.emplace_back(static_cast<float>(width * 0.5), static_cast<float>(height * 0.5));

    std::string label = "object";
    if (!detection.results.empty()) {
      const auto &hypothesis = detection.results.front().hypothesis;
      int class_id = 0;
      std::from_chars(hypothesis.class_id.data(), hypothesis.class_id.data() + hypothesis.class_id.size(), class_id);
      const char *class_name = class_names[static_cast<size_t>(class_id)];
      label = fmt::format("{} {:.4f}", class_name, hypothesis.score);
    }
    labels.emplace_back(std::move(label));
  }

  if (centers.empty()) {
    stream.log(entity_path, rerun::Clear::FLAT);
    return;
  }

  auto boxes = rerun::Boxes2D::from_centers_and_half_sizes(centers, half_sizes)
                   .with_labels(labels)
                   .with_show_labels(rerun::components::ShowLabels(true));
  stream.log(entity_path, boxes);
}

rclcpp::Clock::SharedPtr create_clock(rclcpp::Node &node) {
  bool use_sim_time = false;
  node.get_parameter_or("use_sim_time", use_sim_time, false);
  if (use_sim_time) {
    return node.get_clock(); // reads from /clock topic when sim time enabled
  }
  return std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
}

} // namespace object_tracker::metrics
