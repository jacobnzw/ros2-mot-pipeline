#include "yolo_detector/onnx_model.hpp"

#include <fmt/ranges.h>
#include <opencv2/imgcodecs.hpp>

#include <exception>
#include <iostream>

/**
 * Load an ONNX model and optionally run one image through it.
 * @param argc Number of command-line arguments.
 * @param argv Arguments: executable, model path, and optional image path.
 * @return Zero on success, two for invalid input paths/usage, one on ORT errors.
 */
int main(int argc, char **argv) {
  // Require a model path, with one optional image path for an inference run.
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: onnx_probe <model.onnx> [image]" << std::endl;
    return 2;
  }

  try {
    // Constructing OnnxModel creates an ORT session and loads the graph once.
    yolo_detector::OnnxModel model(argv[1]);
    std::cout << "Loaded model: " << argv[1] << std::endl;

    if (argc == 2) {
      std::cout << "Pass an image path to run inference." << std::endl;
      return 0;
    }

    // OpenCV decodes the image file into a BGR matrix for the model wrapper.
    const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (image.empty()) {
      std::cerr << "Could not read image: " << argv[2] << std::endl;
      return 2;
    }

    // infer() preprocesses the image and invokes the already-loaded ORT session.
    const auto inference = model.infer(image);
    std::vector<std::vector<int64_t>> output_shapes;
    output_shapes.reserve(inference.outputs.size());
    for (const auto &output : inference.outputs) {
      output_shapes.push_back(output.shape);
    }
    std::cout << fmt::format("{}", output_shapes) << std::endl;

  } catch (const std::exception &error) {
    // ORT and OpenCV report failures as exceptions; convert them to a useful
    // message and nonzero process status for shell scripts and ROS launch tools.
    std::cerr << "Inference failed: " << error.what() << std::endl;
    return 1;
  }

  return 0;
}