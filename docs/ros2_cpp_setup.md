# ROS 2 C++ Setup

The `src/object_tracker` package uses `ament_cmake` and `colcon`. OpenCV is provided by Ubuntu/ROS packages. ONNX
Runtime C++ is installed separately because the Python `onnxruntime` dependency does not provide its C++ headers and
libraries.

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

The project helper script below adds these directories to the runtime path and discovers the WSL host-driver directory
when running under WSL.

This setup covers the C++ detector runtime. The project Python environment may have its own CUDA dependencies for
PyTorch; do not remove those when cleaning up old ONNX Runtime libraries.

The SDK prefix must contain `include/onnxruntime_cxx_api.h` (or
`include/onnxruntime/core/session/onnxruntime_cxx_api.h`) and `lib/libonnxruntime.so`. Keep the SDK version consistent
when building and running.

## Build and run

Run these commands from the repository root in zsh. The helper sources ROS 2, sets the ONNX Runtime/cuDNN paths, adds
the runtime library directories, and sources the workspace overlay when it exists. CUDA is enabled by default, so no
CUDA CMake flag is needed:

```bash
source scripts/ros2_env.zsh
test -f "$ONNXRUNTIME_ROOT/lib/libonnxruntime_providers_cuda.so" || {
  echo "CUDA provider not found under ONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT" >&2
  exit 1
}
colcon build --base-paths src --packages-select object_tracker --symlink-install \
  --cmake-clean-cache \
  --cmake-args \
    "-DPython3_EXECUTABLE=/usr/bin/python3" \
    "-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"
source install/setup.zsh

# Load the ONNX model and confirm that ONNX Runtime can create a session.
ros2 run object_tracker onnx_probe models/yolo/yolo11n.onnx

# Run one image through the model and print output tensor shapes.
ros2 run object_tracker onnx_probe \
  models/yolo/yolo11n.onnx \
  data/kitti/left_color/training/image_02/0019/000000.png
```

For CPU-only execution, keep the same CUDA-capable SDK and explicitly disable its CUDA provider:

```bash
colcon build --base-paths src --packages-select object_tracker --symlink-install \
  --cmake-clean-cache \
  --cmake-args \
    "-DPython3_EXECUTABLE=/usr/bin/python3" \
    "-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT" \
    "-DYOLO_ENABLE_CUDA=OFF"
```

With CUDA enabled, CMake requires the ONNX Runtime CUDA provider and session creation fails explicitly if its required
NVIDIA libraries are unavailable; it does not silently switch to CPU-only execution. Use `--cmake-clean-cache` when
switching between GPU and CPU-only builds so CMake reconfigures the provider setting.

In every terminal that launches ROS nodes, source the helper once to set up ROS, the workspace overlay, and CUDA
libraries. This also works from a new terminal; no separate `CUDNN_ROOT`, `LD_LIBRARY_PATH`, or WSL-driver discovery
commands are needed:

```bash
source scripts/ros2_env.zsh
ros2 launch launch/detector_foxglove_test.launch.py
```

For repeated MCAP playback with the detector/tracker test launch, enable its `loop` argument: (optionally combined with
a set process `timeout`)

```bash
timeout 10m ros2 launch launch/detector_test.launch.py loop:=true label:=loop-test
```

With `loop:=true`, playback continues indefinitely; stop it with Ctrl+C. With `loop:=false`, the launch shuts down
automatically when bag playback ends, so it can be invoked sequentially from a script without waiting for Ctrl+C. The
launch defaults to sequence `0019` under `data/kitti/left_color/training/image_02_mcap`. Select another sequence or MCAP
root with ROS 2 launch arguments (which use `name:=value` syntax rather than `--name` flags):

```bash
ros2 launch launch/detector_test.launch.py seq:=0020 \
  kitti_root:=/path/to/image_02_mcap \
  label:=baseline
```

To run every sequence directory in an MCAP root sequentially, use the batch runner:

```bash
uv run scripts/eval/run_kitti_sequences.py \
  --thread-num 1 --precision fp16 --label baseline
```

`--kitti-root` defaults to `data/kitti/left_color/training/image_02_mcap`; each sequence folder must contain
`<sequence>/<sequence>.mcap`. Optionally pass `--seq 0019,0020` to run only the listed sequences. The batch runner
detects Jetson or RTX 3090 hardware, writes `runs/<label>/metadata.yaml` with the run configuration, and stops if a
sequence launch fails. The `precision` launch argument is currently informational and does not change the detector model
path.

The C++ SDK fetched by CMake writes Rerun data, but does not install the standalone Rerun viewer/CLI. Install the viewer
separately, for example with `uv tool install rerun-sdk` (see the
[Rerun viewer installation guide](https://rerun.io/docs/getting-started/install-rerun/viewer)). After running the
detector and tracker, their scalar metrics are written to a unique run directory under `runs/`, for example
`runs/baseline/<datetime>_0020/recording.rrd`. `detector_test.launch.py` accepts a `label:=...` argument and creates the
recording directory automatically using the label, timestamp, and selected sequence. Open the resulting file with
`rerun runs/<run-id>/recording.rrd`. The recording includes detector inference and callback intervals, estimated
cumulative frame drops, ByteTrack association latency and active-track count, and image-stamp-to-tracker-output latency.
Frame-drop estimates use the detector's `expected_frame_rate` parameter and the tracker's `frame_rate` parameter. Each
metric is indexed by the input message timestamp on the `ros2_timestamp` timeline so it can be aligned with the input
images. Set the detector's `log_input_image` parameter to `true` to include each compressed input frame in the same
recording as the metrics. This is disabled by default; enabling it duplicates image data and increases the RRD file
size. When enabled, the detector also logs its decoded bounding boxes with class IDs and confidence scores over the
image. Tracker callback interval and ByteTrack association duration are measured with a steady clock, independently of
ROS simulated time. Every Rerun log event is indexed on both `ros2_timestamp` (message/header time) and `steady_elapsed`
(elapsed monotonic time since recording initialization). This preserves source-frame alignment while giving looped MCAP
replays a non-repeating time axis. The same event is logged once with both timeline coordinates.

To export latency, duration, and callback-interval scalar samples to VictoriaMetrics and open Grafana, run:

```bash
scripts/analyse.sh runs/<run-id>/recording.rrd \
  --hw jetson-orin --quant fp16 --seq 0020 --label baseline
```

The script writes `latency.csv` beside the recording and, if TrackEval detailed results are present under the same run
folder, writes `trackeval.csv` there too. It starts the Compose services, waits for them to respond, and imports both
CSVs into VictoriaMetrics. Latency rows have columns `timestamp_ms,run,stage,hardware,precision,sequence,label,value`
and become `latency_ms` samples. The `--hw`, `--precision`, `--seq`, and `--label` options populate the corresponding
metadata labels; they default to empty strings. TrackEval rows have columns
`timestamp_ms,run,hardware,precision,label,tracker,sequence,class,metric,value` and become `trackeval_kpi` samples with
run/hardware/precision/label/tracker/sequence/class/metric labels. TrackEval timestamps use the newest detailed-results
file modification time; score values are ratios in `[0, 1]`, while count metrics remain counts. The run label defaults
to the recording directory name; pass an optional second argument to `analyse.sh` to override it. The stage label comes
from the Rerun entity path (for example, `/detector/inference_latency_ms` becomes `detector_inference`). Counters and
other non-duration scalars are omitted. You can also run
`uv run scripts/eval/extract_pipeline_kpi.py runs/<run-id>/recording.rrd --hw jetson-orin --precision fp16 --seq 0020 --label baseline`
to export only the CSV with those labels.

The exporter uses `steady_elapsed` when present, falling back to `ros2_timestamp` for older recordings. It maps elapsed
time so the final sample is at export time, preserving the spacing between samples and making the recording visible in
Grafana's current time range.

The Compose stack uses VictoriaMetrics v1.155.0 and Grafana 12.0.0. The VictoriaMetrics image is pulled on first
startup. The provisioned `ROS 2 Latency Analysis` dashboard includes a detector-inference heatmap (5-second X buckets
and 25 latency buckets), rolling 5-second p50/p95/p99 inference latency, an overall histogram for a selectable stage,
and a TrackEval KPI table for the selected run. The KPI table includes per-sequence and `COMBINED` values. Select the
imported run in the dashboard's `Run` variable; the `Stage` variable controls the histogram panel. The Prometheus data
source uses a 100 ms minimum query interval to retain approximately the original 10 Hz sample resolution for the heatmap
and histogram.

VictoriaMetrics data and Grafana state persist in Docker named volumes across container restarts. VictoriaMetrics
retains data for 31 days. Services bind to localhost only; Grafana's initial local credentials are `admin` / `admin`.
Change the password in Grafana after the first login; the updated credentials are saved in Grafana's Docker named
volume. `docker compose down -v` deletes the persisted data, including the saved credentials.


To use a system-wide cuDNN installation, set `CUDNN_LIBRARY_DIR` before sourcing the helper. To override the default SDK
or toolkit locations, set `ONNXRUNTIME_ROOT` or `CUDA_ROOT` first.

To export a one-page PDF with histograms for the detector inference latency, detector and tracker callback intervals,
and end-to-end pipeline latency, run this from the repository root:

```bash
uv run scripts/rrd_latency_histogram.py runs/<run-id>/recording.rrd
```

By default, the PDF is written beside the recording as `object_tracker_latency_histogram.pdf`. Pass `--output path.pdf`
to choose a different output path.

The probe letterboxes the image, converts it to RGB float values in `[0, 1]`, and runs the model. It reports tensor
shapes but does not decode detections or apply NMS; those details depend on the ONNX export configuration. `build/`,
`install/`, and `log/` are ignored by git.