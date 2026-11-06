#pragma once

#include <array>
#include <string_view>

namespace object_tracker {

inline constexpr std::array<const char *, 80> kCocoClassNames{
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",           "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

struct CocoKittiClassMapping {
  int coco_class_id;
  std::string_view kitti_class_name;
};

inline constexpr std::array<CocoKittiClassMapping, 6> kCocoToKittiClasses{{
    {0, "Pedestrian"},
    {1, "Cyclist"},
    {2, "Car"},
    {5, "Car"},
    {6, "Tram"},
    {7, "Truck"},
}};

inline constexpr std::array<std::string_view, 5> kKittiClassNames{
    "Pedestrian", "Car", "Cyclist", "Tram", "Truck",
};

constexpr std::string_view kitti_class_for_coco(int coco_class_id) {
  for (const auto &mapping : kCocoToKittiClasses) {
    if (mapping.coco_class_id == coco_class_id) {
      return mapping.kitti_class_name;
    }
  }
  return {};
}

} // namespace object_tracker
