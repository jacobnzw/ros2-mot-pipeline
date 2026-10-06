#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace yolo_detector
{

/**
 * Loads an ONNX model once and runs single-image inference with ONNX Runtime.
 * The ORT environment and session are kept as members so they outlive every
 * inference call made through this object.
 */
class OnnxModel
{
public:
  /**
   * Create an ORT session from an ONNX model file.
   * @param model_path Filesystem path to the exported model.
   */
  explicit OnnxModel(const std::string & model_path);

  /**
   * Preprocess one OpenCV BGR image, run the session, and return output shapes.
   * @param bgr_image Input image in OpenCV's usual BGR channel order.
   * @return One dimension vector for each model output tensor.
   */
  std::vector<std::vector<int64_t>> infer(const cv::Mat & bgr_image);

private:
  // Env owns process-level ORT state and must outlive sessions created from it.
  Ort::Env env_;
  // Options configure how ORT creates and optimizes the model session.
  Ort::SessionOptions session_options_;
  // Session owns the loaded graph and is reused across frames to avoid reload cost.
  Ort::Session session_;

  // ORT Run() takes tensor names as C strings; retain their copied values here.
  std::string input_name_;
  std::vector<std::string> output_names_;
};

}  // namespace yolo_detector