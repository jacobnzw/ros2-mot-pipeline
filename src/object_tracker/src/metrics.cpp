#include "object_tracker/metrics.hpp"

#include <rerun.hpp>

#include <mutex>
#include <stdexcept>
#include <string>

namespace object_tracker::metrics {
namespace {

rerun::RecordingStream &recording_stream() {
  static rerun::RecordingStream stream("ros2-mot-pipeline");
  return stream;
}

} // namespace

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

void log_scalar(std::string_view entity_path, double value) {
  rerun::RecordingStream::current().log(entity_path, rerun::Scalars(value));
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
