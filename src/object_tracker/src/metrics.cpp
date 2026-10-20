#include "object_tracker/metrics.hpp"

#include <rerun.hpp>

#include <mutex>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>

namespace object_tracker::metrics {
namespace {

// TODO: why like this? static?
rerun::RecordingStream &recording_stream() {
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

void log_encoded_image(std::string_view entity_path, std::span<const uint8_t> image_data, std::string_view format,
                       int64_t ros_timestamp_ns) {
  if (image_data.empty()) {
    throw std::invalid_argument("Cannot log an empty encoded image");
  }

  std::string normalized_format(format);
  std::transform(normalized_format.begin(), normalized_format.end(), normalized_format.begin(),
                 [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
  std::optional<rerun::MediaType> media_type;
  if (normalized_format.find("jpeg") != std::string::npos || normalized_format.find("jpg") != std::string::npos) {
    media_type = rerun::MediaType::jpeg();
  } else if (normalized_format.find("png") != std::string::npos) {
    media_type = rerun::MediaType::png();
  }

  auto &stream = rerun::RecordingStream::current();
  stream.set_time_timestamp_nanos_since_epoch("ros2_timestamp", ros_timestamp_ns);
  const auto encoded_image = rerun::EncodedImage::from_bytes(
      rerun::Collection<uint8_t>::borrow(image_data.data(), image_data.size()), media_type);
  stream.log(entity_path, encoded_image);
}

rclcpp::Clock::SharedPtr create_clock(rclcpp::Node &node) {
  bool use_sim_time = false;
  node.get_parameter_or("use_sim_time", use_sim_time, false);
  if (use_sim_time) {
    return node.get_clock();
  }
  return std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
}

} // namespace object_tracker::metrics
