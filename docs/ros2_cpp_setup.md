# ROS 2 C++ Setup

The `src/yolo_detector` package uses `ament_cmake` and `colcon`. OpenCV is provided by Ubuntu/ROS packages. ONNX Runtime
C++ is installed separately because the Python `onnxruntime` dependency does not provide its C++ headers and libraries.

## System dependencies

These commands are for Ubuntu 24.04 with ROS 2 Jazzy in zsh. For Bash, replace `setup.zsh` with `setup.bash`. Initialize
rosdep once per machine; skip `rosdep init` if it has already been initialized.

```bash
sudo apt update
sudo apt install ros-jazzy-ros-base ros-dev-tools libopencv-dev
sudo rosdep init
rosdep update
source /opt/ros/jazzy/setup.zsh
rosdep install --from-paths src --ignore-src -r -y
```

## ONNX Runtime C++

Use one CUDA-capable ONNX Runtime C++ SDK for both GPU and CPU execution. The CUDA 12 build of ONNX Runtime 1.30.0 works
with the system CUDA 12.9 toolkit and cuDNN 9; the SDK also includes the CPU execution provider. CUDA is enabled by
default. To build CPU-only, pass `-DYOLO_ENABLE_CUDA=OFF` using the same SDK.

On WSL, `nvitop` reports the driver CUDA API level, not the installed toolkit version. Check
`readlink -f /usr/local/cuda` and `nvcc --version`; this setup expects CUDA 12.x (tested with 12.9).

Download the CUDA 12 ONNX Runtime SDK and install the cuDNN 9 runtime locally. The CUDA toolkit provides CUDA runtime
and cuBLAS libraries; this installs only cuDNN and does not replace the system toolkit:

```bash
ONNXRUNTIME_VERSION=1.30.0
ONNXRUNTIME_ARCHIVE="onnxruntime-linux-x64-gpu_cuda12-${ONNXRUNTIME_VERSION}.tgz"
curl -fL \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ARCHIVE}" \
  -o "/tmp/${ONNXRUNTIME_ARCHIVE}"
mkdir -p "$HOME/opt"
tar -xzf "/tmp/${ONNXRUNTIME_ARCHIVE}" -C "$HOME/opt"
export ONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-linux-x64-gpu_cuda12-${ONNXRUNTIME_VERSION}"

export CUDNN_ROOT="$HOME/opt/cuda12-cudnn-runtime"
uv pip install --python .venv/bin/python --target "$CUDNN_ROOT" --no-deps \
  nvidia-cudnn-cu12==9.27.0.42
ln -sfn libcudnn.so.9 "$CUDNN_ROOT/nvidia/cudnn/lib/libcudnn.so"
```

Here `uv` is only the installer: `--target "$CUDNN_ROOT"` puts cuDNN under `~/opt`, not in `.venv` and not system-wide.
The wheel provides the versioned `libcudnn.so.9`, while ONNX Runtime looks up `libcudnn.so`, so the symlink supplies the
name it expects. This keeps the C++ runtime dependency isolated from both the project Python environment and OS
packages. A system-wide cuDNN 9 installation is also valid; install its runtime and development packages so the
unversioned `libcudnn.so` name is available, then point the dynamic linker at that library directory instead and omit
this local installation.

The GPU SDK must contain `lib/libonnxruntime_providers_cuda.so`; CMake checks for it when CUDA is enabled. Keep
`ONNXRUNTIME_ROOT` pointed at this SDK for both build modes. Do not commit the extracted SDK or cuDNN libraries to the
repository.

For WSL, put the matching host-driver directory first in the runtime library path; an older distro-installed NVIDIA PTX
JIT library can conflict with the WSL driver:

```bash
WSL_NVIDIA_DRIVER_ROOT="$(dirname "$(find /usr/lib/wsl/drivers \
  -name libnvidia-ptxjitcompiler.so.1 -print -quit)")"
export LD_LIBRARY_PATH="$WSL_NVIDIA_DRIVER_ROOT:$CUDNN_ROOT/nvidia/cudnn/lib:/usr/local/cuda/lib64:$ONNXRUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For native Ubuntu rather than WSL, omit `WSL_NVIDIA_DRIVER_ROOT` and use the NVIDIA driver libraries provided by the
host.

This setup covers the C++ detector runtime. The project Python environment may have its own CUDA dependencies for
PyTorch; do not remove those when cleaning up old ONNX Runtime libraries.

The SDK prefix must contain `include/onnxruntime_cxx_api.h` (or
`include/onnxruntime/core/session/onnxruntime_cxx_api.h`) and `lib/libonnxruntime.so`. Keep the SDK version consistent
when building and running.

## Build and run

Run these commands from the repository root in a shell with `ONNXRUNTIME_ROOT` and the runtime library path set as
above. CUDA is enabled by default, so no CUDA CMake flag is needed:

```bash
source /opt/ros/jazzy/setup.zsh
export ONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-linux-x64-gpu_cuda12-1.30.0"
test -f "$ONNXRUNTIME_ROOT/lib/libonnxruntime_providers_cuda.so" || {
  echo "CUDA provider not found under ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT" >&2
  exit 1
}
colcon build --base-paths src --packages-select yolo_detector --symlink-install \
  --cmake-clean-cache \
  --cmake-args \
    "-DPython3_EXECUTABLE=/usr/bin/python3" \
    "-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
source install/setup.zsh

# Load the ONNX model and confirm that ONNX Runtime can create a session.
ros2 run yolo_detector onnx_probe models/yolo/yolo11n.onnx

# Run one image through the model and print output tensor shapes.
ros2 run yolo_detector onnx_probe \
  models/yolo/yolo11n.onnx \
  data/kitti/left_color/training/image_02/0019/000000.png
```

For CPU-only execution, keep the same CUDA-capable SDK and explicitly disable its CUDA provider:

```bash
colcon build --base-paths src --packages-select yolo_detector --symlink-install \
  --cmake-clean-cache \
  --cmake-args \
    "-DPython3_EXECUTABLE=/usr/bin/python3" \
    "-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT" \
    "-DYOLO_ENABLE_CUDA=OFF"
```

With CUDA enabled, CMake requires the ONNX Runtime CUDA provider and session creation fails explicitly if its required
NVIDIA libraries are unavailable; it does not silently switch to CPU-only execution. Use `--cmake-clean-cache` when
switching between GPU and CPU-only builds so CMake reconfigures the provider setting.

The process that runs `ros2 launch` must also have the CUDA, cuDNN, and ONNX Runtime directories in its library search
path. Shell exports do not carry over to a different terminal or to a VS Code launch session. In WSL, set these in the
same terminal where you launch the detector:

```bash
source /opt/ros/jazzy/setup.zsh
source install/setup.zsh
export ONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-linux-x64-gpu_cuda12-1.30.0"
export CUDNN_ROOT="$HOME/opt/cuda12-cudnn-runtime"
WSL_NVIDIA_DRIVER_ROOT="$(dirname "$(find /usr/lib/wsl/drivers \
  -name libnvidia-ptxjitcompiler.so.1 -print -quit)")"
export LD_LIBRARY_PATH="$WSL_NVIDIA_DRIVER_ROOT:$CUDNN_ROOT/nvidia/cudnn/lib:/usr/local/cuda/lib64:$ONNXRUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ros2 launch launch/detector_foxglove_test.launch.py
```

If launching from VS Code or another terminal, configure the same `LD_LIBRARY_PATH` for that launch environment. For a
system-wide cuDNN installation, use its library directory in place of `$CUDNN_ROOT/nvidia/cudnn/lib`.

The probe letterboxes the image, converts it to RGB float values in `[0, 1]`, and runs the model. It reports tensor
shapes but does not decode detections or apply NMS; those details depend on the ONNX export configuration. `build/`,
`install/`, and `log/` are ignored by git.