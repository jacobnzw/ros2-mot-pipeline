#include "yolo_detector/yolo_detector_component.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace yolo_detector {
namespace {

constexpr std::array<const char *, 80> kCocoClassNames{
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

struct DetectionCandidate {
  int class_id;
  float score;
  cv::Rect2d box;
};

std::vector<vision_msgs::msg::Detection2D> decode_yolo_output(const InferenceResult &inference,
                                                              const sensor_msgs::msg::Image &image,
                                                              double confidence_threshold,
                                                              double nms_threshold) {
  if (inference.outputs.size() != 1) {
    throw std::runtime_error("YOLO decoder expects exactly one output tensor");
  }
  const auto &output = inference.outputs.front();
  constexpr int64_t feature_count = 4 + static_cast<int64_t>(kCocoClassNames.size());
  if (output.shape.size() != 3 || output.shape[0] != 1) {
    throw std::runtime_error("Expected YOLO output shape [1, 84, N] or [1, N, 84]");
  }

  const bool channels_first = output.shape[1] == feature_count;
  const bool channels_last = output.shape[2] == feature_count;
  if (!channels_first && !channels_last) {
    throw std::runtime_error("Expected 84 YOLO features (4 box values and 80 COCO scores)");
  }
  const int64_t candidate_count = channels_first ? output.shape[2] : output.shape[1];
  const auto expected_value_count = static_cast<size_t>(output.shape[1] * output.shape[2]);
  if (candidate_count <= 0 || output.values.size() != expected_value_count ||
      inference.scale <= 0.0F || inference.original_width <= 0 || inference.original_height <= 0) {
    throw std::runtime_error("YOLO output tensor or letterbox metadata is invalid");
  }

  const auto value_at = [&output, channels_first, candidate_count](int64_t feature,
                                                                   int64_t candidate) {
    const size_t index = channels_first ? static_cast<size_t>(feature * candidate_count + candidate)
                                        : static_cast<size_t>(candidate * 84 + feature);
    return output.values[index];
  };

  std::vector<DetectionCandidate> candidates;
  for (int64_t candidate = 0; candidate < candidate_count; ++candidate) {
    int class_id = 0;
    float score = value_at(4, candidate);
    for (int class_index = 1; class_index < static_cast<int>(kCocoClassNames.size());
         ++class_index) {
      const float class_score = value_at(4 + class_index, candidate);
      if (class_score > score) {
        score = class_score;
        class_id = class_index;
      }
    }
    if (!std::isfinite(score) || score < confidence_threshold) {
      continue;
    }

    const double center_x = value_at(0, candidate);
    const double center_y = value_at(1, candidate);
    const double width = value_at(2, candidate);
    const double height = value_at(3, candidate);
    if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(width) ||
        !std::isfinite(height) || width <= 0.0 || height <= 0.0) {
      continue;
    }

    const double x0 = std::clamp((center_x - width * 0.5 - inference.pad_left) / inference.scale,
                                 0.0, static_cast<double>(inference.original_width));
    const double y0 = std::clamp((center_y - height * 0.5 - inference.pad_top) / inference.scale,
                                 0.0, static_cast<double>(inference.original_height));
    const double x1 = std::clamp((center_x + width * 0.5 - inference.pad_left) / inference.scale,
                                 0.0, static_cast<double>(inference.original_width));
    const double y1 = std::clamp((center_y + height * 0.5 - inference.pad_top) / inference.scale,
                                 0.0, static_cast<double>(inference.original_height));
    if (x1 <= x0 || y1 <= y0) {
      continue;
    }
    candidates.push_back({class_id, score, cv::Rect2d(x0, y0, x1 - x0, y1 - y0)});
  }

  std::vector<vision_msgs::msg::Detection2D> detections;
  for (int class_id = 0; class_id < static_cast<int>(kCocoClassNames.size()); ++class_id) {
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<size_t> candidate_indices;
    for (size_t index = 0; index < candidates.size(); ++index) {
      if (candidates[index].class_id == class_id) {
        boxes.push_back(candidates[index].box);
        scores.push_back(candidates[index].score);
        candidate_indices.push_back(index);
      }
    }

    std::vector<int> retained_indices;
    cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(confidence_threshold),
                      static_cast<float>(nms_threshold), retained_indices);

    for (const int retained_index : retained_indices) {
      const auto &candidate = candidates[candidate_indices.at(static_cast<size_t>(retained_index))];

      vision_msgs::msg::Detection2D detection;
      detection.header = image.header;
      detection.bbox.center.position.x = candidate.box.x + candidate.box.width * 0.5;
      detection.bbox.center.position.y = candidate.box.y + candidate.box.height * 0.5;
      detection.bbox.center.theta = 0.0;
      detection.bbox.size_x = candidate.box.width;
      detection.bbox.size_y = candidate.box.height;

      vision_msgs::msg::ObjectHypothesisWithPose result;
      result.hypothesis.class_id = kCocoClassNames[class_id];
      result.hypothesis.score = candidate.score;

      detection.results.push_back(std::move(result));
      detections.push_back(std::move(detection));
    }
  }

  std::sort(detections.begin(), detections.end(), [](const auto &left, const auto &right) {
    return left.results.front().hypothesis.score > right.results.front().hypothesis.score;
  });
  return detections;
}

} // namespace

YoloDetectorComponent::YoloDetectorComponent(const rclcpp::NodeOptions &options)
    : Node("yolo_detector_component", options) {

  // Input parameters modifiable during: ros2 component load
  // Example: -p model_path:=model/yolo/yolo11n.onnx -p image_topic:=/camera/image_raw
  declare_parameter("model_path", std::string(""));
  declare_parameter("image_topic", std::string("/image_raw"));
  declare_parameter("detection_topic", std::string("~/detections"));
  declare_parameter("confidence_threshold", 0.25);
  declare_parameter("nms_threshold", 0.45);

  confidence_threshold_ = get_parameter("confidence_threshold").as_double();
  nms_threshold_ = get_parameter("nms_threshold").as_double();
  if (confidence_threshold_ < 0.0 || confidence_threshold_ > 1.0 || nms_threshold_ < 0.0 ||
      nms_threshold_ > 1.0) {
    throw std::invalid_argument("confidence_threshold and nms_threshold must be in [0, 1]");
  }

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

void YoloDetectorComponent::imageCallback(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
  if (!model_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Detector model is not loaded.");
    return;
  }

  try {
    // Model inference: run the incomming image through the YOLO detector
    const auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    const auto inference = model_->infer(cv_ptr->image);

    vision_msgs::msg::Detection2DArray detections;
    detections.header = msg->header;
    detections.detections =
        decode_yolo_output(inference, *msg, confidence_threshold_, nms_threshold_);
    RCLCPP_DEBUG(get_logger(), "Decoded %zu detections", detections.detections.size());

    detections_pub_->publish(detections);

  } catch (const std::exception &ex) {
    RCLCPP_ERROR(get_logger(), "Inference failed: %s", ex.what());
  }
}

} // namespace yolo_detector

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_detector::YoloDetectorComponent)
