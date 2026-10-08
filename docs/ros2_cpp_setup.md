# ROS 2 C++ Setup

The `src/yolo_detector` package uses `ament_cmake` and `colcon`. OpenCV is provided by Ubuntu/ROS packages. ONNX Runtime C++ is installed separately because the Python `onnxruntime` dependency does not provide its C++ headers and libraries.

## System dependencies

These commands are for Ubuntu 24.04 with ROS 2 Jazzy in zsh. For Bash, replace `setup.zsh` with `setup.bash`. Initialize rosdep once per machine; skip `rosdep init` if it has already been initialized.

```bash
sudo apt update
sudo apt install ros-jazzy-ros-base ros-dev-tools libopencv-dev
sudo rosdep init
rosdep update
source /opt/ros/jazzy/setup.zsh
rosdep install --from-paths src --ignore-src -r -y
```

## ONNX Runtime C++

The Python dependencies in `pyproject.toml` require ONNX Runtime 1.30.0 or newer; use 1.30.0 for this C++ probe so the CPU runtime version is explicit and aligned. These commands download the x86-64 CPU archive from the official ONNX Runtime releases. Verify the SHA-256 against the release's published checksum before extracting it outside the repository:

```bash
ONNXRUNTIME_VERSION=1.30.0
ONNXRUNTIME_ARCHIVE="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
curl -fL \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ARCHIVE}" \
  -o "/tmp/${ONNXRUNTIME_ARCHIVE}"
printf '%s  %s\n' \
  'a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd' \
  "/tmp/${ONNXRUNTIME_ARCHIVE}" | sha256sum -c -
mkdir -p "$HOME/opt"
tar -xzf "/tmp/${ONNXRUNTIME_ARCHIVE}" -C "$HOME/opt"
export ONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"
export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The prefix must contain `include/onnxruntime_cxx_api.h` (or `include/onnxruntime/core/session/onnxruntime_cxx_api.h`) and `lib/libonnxruntime.so`. Keep `ONNXRUNTIME_ROOT` and the exact release version consistent when building and running. Do not commit the extracted SDK to this repository.

## Build and run

Run these commands from the repository root in a shell where `ONNXRUNTIME_ROOT` is set:

```bash
source /opt/ros/jazzy/setup.zsh
colcon build --base-paths src --packages-select yolo_detector --symlink-install \
  --cmake-args "-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
source install/setup.zsh

# Load the ONNX model and confirm that ONNX Runtime can create a session.
ros2 run yolo_detector onnx_probe models/yolo/yolo11n.onnx

# Run one image through the model and print output tensor shapes.
ros2 run yolo_detector onnx_probe \
  models/yolo/yolo11n.onnx \
  data/kitti/left_color/training/image_02/0019/000000.png
```

The probe letterboxes the image, converts it to RGB float values in `[0, 1]`, and runs the model. It reports tensor shapes but does not decode detections or apply NMS; those details depend on the ONNX export configuration. `build/`, `install/`, and `log/` are ignored by git.