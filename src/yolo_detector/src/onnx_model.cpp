#include "yolo_detector/onnx_model.hpp"

#include <fmt/ranges.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolo_detector {
namespace {

/**
 * Configure graph optimizations before creating an ORT session.
 * SessionOptions are consumed when the session is constructed; they do not
 * hold the model or perform inference themselves.
 */
Ort::SessionOptions make_session_options() {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return options;
}

} // namespace

/**
 * Initialize ORT's shared environment, configure the session, and load the
 * model. The member declaration order in the header ensures env_ exists before
 * session_ is constructed from it.
 *
 * Ort::Env is the process-wide runtime. It owns shared infrastructure: thread
 * pools, logging, and (optionally) shared allocators. One per process.
 *
 * Ort::Session is created from an Ort::Env — the env is passed as the first
 * constructor argument. It loads a specific model, applies graph optimizations,
 * and executes inference.
 *
 * - If env is destroyed, all sessions referencing it become invalid.
 * - 1 `Env` -> N `Session`s (different models, different `SessionOptions`,
 * etc.)
 * - `Env` manages thread pools; sessions inherit thread behavior from it.
 *    Allocators registered on the env can be shared across sessions.
 * - `SessionOptions` are per-session — 2 sessions on the same env can use
 * different providers (CPU vs. GPU), different optimization levels, etc.
 */
OnnxModel::OnnxModel(const std::string &model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo_detector"), session_options_(make_session_options()),
      session_(env_, model_path.c_str(), session_options_) {
  if (session_.GetInputCount() != 1) {
    throw std::runtime_error("Expected a model with exactly one input tensor");
  }

  // ORT allocates the name returned here. The allocated wrapper frees it when
  // it leaves scope, so copy the text into input_name_ before that happens.
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session_.GetInputNameAllocated(0, allocator);
  input_name_ = input_name.get();

  // Cache all output names once; inference only needs to pass these names back
  // to the session and should not query model metadata for every image.
  const size_t output_count = session_.GetOutputCount();
  if (output_count == 0) {
    throw std::runtime_error("Model has no output tensors");
  }
  output_names_.reserve(output_count);
  for (size_t index = 0; index < output_count; ++index) {
    auto output_name = session_.GetOutputNameAllocated(index, allocator);
    output_names_.emplace_back(output_name.get());
  }
}

/**
 * Run one image through the loaded model and return its output tensor shapes.
 * This probe intentionally does not interpret YOLO values or apply NMS.
 */
std::vector<std::vector<int64_t>> OnnxModel::infer(const cv::Mat &bgr_image) {
  if (bgr_image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  // TypeInfo owns model metadata. Keep it alive while using the tensor-shape
  // view returned from it; otherwise that view would refer to released state.
  const auto input_type_info = session_.GetInputTypeInfo(0);
  const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
  auto input_shape = input_info.GetShape();

  // This first probe supports the common YOLO image input: one NCHW float
  // tensor with three color channels. Reject other layouts rather than silently
  // feeding them incorrectly.
  if (input_shape.size() != 4 || (input_shape[0] > 0 && input_shape[0] != 1) ||
      (input_shape[1] > 0 && input_shape[1] != 3)) {

    const auto shapes_text = fmt::format("{}", input_shape);
    throw std::runtime_error("Expected a single-image NCHW model input with "
                             "three channels; received " +
                             shapes_text);
  }

  // A negative model dimension ==> dynamic input size.
  // Use 640 as fallback for a dynamic height or width.
  const int height = input_shape[2] > 0 ? static_cast<int>(input_shape[2]) : 640;
  const int width = input_shape[3] > 0 ? static_cast<int>(input_shape[3]) : 640;
  input_shape = {1, 3, height, width};

  const float scale = std::min(static_cast<float>(width) / bgr_image.cols,
                               static_cast<float>(height) / bgr_image.rows);
  const int resized_width = static_cast<int>(std::round(bgr_image.cols * scale));
  const int resized_height = static_cast<int>(std::round(bgr_image.rows * scale));

  // Letterboxing
  // Preserve aspect ratio, then pad to the model's exact dimensions.
  // Avoids stretching objects while satisfying model's fixed inputs.
  // Create blank canvas with "uninformative" color (114, 114, 114)
  // then paste the resized image centered on the canvas
  cv::Mat resized;
  cv::resize(bgr_image, resized, cv::Size(resized_width, resized_height));
  cv::Mat letterboxed(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
  const int left = (width - resized_width) / 2;
  const int top = (height - resized_height) / 2;
  resized.copyTo(letterboxed(cv::Rect(left, top, resized_width, resized_height)));

  // The image reader returns BGR bytes; exported ONNX model expects RGB float
  // values normalized to [0, 1]. OpenCV stores these pixels as interleaved HWC
  // data.
  cv::Mat input_blob;
  cv::dnn::blobFromImage(letterboxed, input_blob,
                         1.0 / 255.0, // scale pixel values to [0, 1]
                         cv::Size(),  // keep the existing image size
                         cv::Scalar(),
                         true,  // swap BGR to RGB
                         false, // don't crop
                         CV_32F);

  // CreateCpu builds an ORT memory descriptor for CPU memory (device "Cpu",
  // id 0). It describes memory to ORT; it does not allocate the vector.
  // OrtArenaAllocator selects the allocator category recorded in that
  // descriptor. It does not move input_tensor_values into ORT's arena.
  // OrtMemTypeDefault selects the execution provider's default memory type.
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // Wrap caller-owned data as a float tensor; it does not copy
  // or take ownership of the vector's storage. The shape describes those
  // floats as [batch, channels, height, width] (NCHW).
  auto input_tensor =
      Ort::Value::CreateTensor<float>(memory_info, input_blob.ptr<float>(), input_blob.total(),
                                      input_shape.data(), input_shape.size());

  // Run() accepts arrays of input/output names and tensors. The session matches
  // each name to the corresponding graph input or output and executes the
  // graph.
  std::vector<const char *> output_name_pointers;
  output_name_pointers.reserve(output_names_.size());
  for (const auto &name : output_names_) {
    output_name_pointers.push_back(name.c_str());
  }

  // RUN INFERENCE
  const char *input_name = input_name_.c_str();
  auto outputs = session_.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1,
                              output_name_pointers.data(), output_name_pointers.size());

  // Return only dimensions for now, while the actual output values remain owned
  // by ORT's output tensors. A detector will later decode those values to
  // boxes.
  std::vector<std::vector<int64_t>> output_shapes;
  output_shapes.reserve(outputs.size());
  for (const auto &output : outputs) {
    output_shapes.push_back(output.GetTensorTypeAndShapeInfo().GetShape());
  }
  return output_shapes;
}

} // namespace yolo_detector