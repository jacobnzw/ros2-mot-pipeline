#include "object_tracker/tracker/tracker_component.hpp"
#include "object_tracker/tracker/detection.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace object_tracker {
// TODO: rename to bytetrack_component: since this is bytetrack not a generic tracker implementation here
TrackerComponent::TrackerComponent(const rclcpp::NodeOptions &options) : Node("tracker_component", options) {
  declare_parameter("detections_topic", std::string("/detections"));
  declare_parameter("tracks_topic", std::string("~/tracks"));
  declare_parameter("high_threshold", 0.25);      // detections above this take part in the first association
  declare_parameter("low_threshold", 0.1);        // (low, high) detections only rescue existing tracks
  declare_parameter("new_track_threshold", 0.25); // minimum score to start a new track
  declare_parameter("match_threshold", 0.8);      // max (1 - IoU) cost in the first association
  declare_parameter("track_buffer", 10);          // frames a lost track is kept for re-identification
  declare_parameter("frame_rate", 10.0);
  declare_parameter("fuse_score", true);

  ByteTrackerOptions tracker_options;
  tracker_options.high_threshold = static_cast<float>(get_parameter("high_threshold").as_double());
  tracker_options.low_threshold = static_cast<float>(get_parameter("low_threshold").as_double());
  tracker_options.new_track_threshold = static_cast<float>(get_parameter("new_track_threshold").as_double());
  tracker_options.match_threshold = static_cast<float>(get_parameter("match_threshold").as_double());
  tracker_options.track_buffer = get_parameter("track_buffer").as_int();
  tracker_options.frame_rate = static_cast<float>(get_parameter("frame_rate").as_double());
  tracker_options.fuse_score = get_parameter("fuse_score").as_bool();
  tracker_ = ByteTracker(tracker_options);

  const auto detections_topic = get_parameter("detections_topic").as_string();
  const auto qos_sub = rclcpp::SensorDataQoS().keep_last(1);
  detections_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
      detections_topic, qos_sub,
      [this](const vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) { this->detectionsCallback(msg); });

  const auto tracks_topic = get_parameter("tracks_topic").as_string();
  const auto qos_pub = rclcpp::SensorDataQoS().keep_last(1);
  tracks_pub_ = create_publisher<object_tracker::msg::TrackedObjectArray>(tracks_topic, qos_pub);
}

void TrackerComponent::detectionsCallback(const vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) {
  auto detections = convert(*msg);

  std::vector<TrackedObject> tracked_objects;
  tracker_.update(detections, tracked_objects);

  auto tracks_out = convert(tracked_objects, msg->header);
  tracks_pub_->publish(tracks_out);
}

std::vector<Detection> TrackerComponent::convert(const vision_msgs::msg::Detection2DArray &detections_msg) const {
  std::vector<Detection> detections;
  detections.reserve(detections_msg.detections.size());

  for (const auto &detection2d : detections_msg.detections) {
    if (detection2d.results.empty()) {
      RCLCPP_WARN(get_logger(), "Skipping a detection without a class hypothesis.");
      continue;
    }

    const auto &hypothesis = detection2d.results.front().hypothesis;
    int class_id = 0;
    const auto &class_id_text = hypothesis.class_id;
    const auto [end, error] =
        std::from_chars(class_id_text.data(), class_id_text.data() + class_id_text.size(), class_id);
    if (error != std::errc{} || end != class_id_text.data() + class_id_text.size()) {
      RCLCPP_WARN(get_logger(), "Skipping detection with non-numeric class ID '%s'.", class_id_text.c_str());
      continue;
    }

    Detection detection;
    detection.box = Box::from_center(
        static_cast<float>(detection2d.bbox.center.position.x), static_cast<float>(detection2d.bbox.center.position.y),
        static_cast<float>(detection2d.bbox.size_x), static_cast<float>(detection2d.bbox.size_y));
    detection.score = static_cast<float>(hypothesis.score);
    detection.class_id = class_id;
    detections.push_back(std::move(detection));
  }

  return detections;
}

msg::TrackedObjectArray TrackerComponent::convert(const std::vector<TrackedObject> &tracked_objects,
                                                  const std_msgs::msg::Header &header) const {
  msg::TrackedObjectArray tracks_msg;
  tracks_msg.header = header;
  tracks_msg.objects.reserve(tracked_objects.size());

  for (const auto &tracked_object : tracked_objects) {
    auto &tracked_object_msg = tracks_msg.objects.emplace_back();
    tracked_object_msg.track_id = tracked_object.track_id;
    tracked_object_msg.score = tracked_object.score;
    tracked_object_msg.class_id = tracked_object.class_id;
    tracked_object_msg.age_frames = tracked_object.age_frames;
    tracked_object_msg.x1 = tracked_object.box.x1;
    tracked_object_msg.y1 = tracked_object.box.y1;
    tracked_object_msg.x2 = tracked_object.box.x2;
    tracked_object_msg.y2 = tracked_object.box.y2;
  }

  return tracks_msg;
}
} // namespace object_tracker

RCLCPP_COMPONENTS_REGISTER_NODE(object_tracker::TrackerComponent)