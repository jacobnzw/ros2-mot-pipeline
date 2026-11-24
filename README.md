# ROS2 Multi-Object Tracking Pipeline — PC vs Jetson

**Latency, throughput and tracking-quality characterisation of a YOLO + ByteTrack ROS2 pipeline**

Modular ROS2 multi-object tracking (MOT) system and an experimental / evaluation harness for measuring how the same
pipeline behaves on a high-end PC versus a Jetson Orin NX 16GB.



## 🎯 Project Goal

Quantify the **latency distributions**, **frame drops**, **throughput** and **tracking quality** of a production-style
ROS2 MOT pipeline when moving from:

- **PC** — RTX 3090 24 GB
- **Jetson** — Orin NX 16 GB

under controlled MCAP replay of real tracking data.

The study examines:

- Platform effect (PC ↔ Jetson at matched precision)
- Precision effect (FP32 vs FP16, especially on Jetson)
- Executor threading (1 / 2 / 4 threads)
- Load / replay-rate effect (1.0× / 2.0× / 3.0×)
- Interaction of load with platform (how drops and tails scale)



## 📌 Status & Scope

- ✅ Full experiment grid completed on both platforms (36 configs × 21 sequences)
- ✅ Metrics pipeline → long-form Parquet (GitHub Release)
- ✅ TrackEval integration
- ❌ Live camera drivers, multi-camera, 3D lifting, online dashboards (out of scope)

This project prioritises **reproducible offline characterisation** of latency, drops, throughput and tracking quality
under realistic load.



## 📄 Report

A detailed write-up with experimental setup, figures and analysis lives in [`docs/report.md`](docs/report.md).



## ✨ Key Features

- **Modular ROS2 components**
  - YOLO detector (ONNX Runtime, FP32 / FP16)
  - ByteTrack tracker
  - Zero-copy intra-process communication
- **Correct timing & measurement**
  - Sensor timestamp propagation
  - Steady-clock latency and callback-duration logging
- **Offline observability**
  - Rerun (`.rrd`) recordings
  - Metrics exported to a long-form Parquet table
  - Plotting scripts (`scripts/plot_results.py`, `scripts/build_parquet.py`)
- **Configurable execution**
  - `keep_last(1)` QoS on all links
  - Single- or Multi-Threaded executor (`n_threads` launch argument)
- **Reproducible experiment grid**
  - All 21 KITTI tracking training sequences
  - Cartesian product of precision × threads × rate



## 🏗️ ROS2 Pipeline Architecture

```text
mcap replay (KITTI)
        │
        ▼
┌───────────────────┐
│  Detector node    │  YOLO (FP32 / FP16)
│  (detections)     │
└─────────┬─────────┘
          │ zero-copy + keep_last(1)
          ▼
┌───────────────────┐
│  Tracker node     │  ByteTrack
│  (tracks)         │
└─────────┬─────────┘
          │
          ▼
   Rerun + metrics → Parquet
```

Both nodes run inside a single composable component container. Measurements use steady clocks; all analysis is offline.

TODO: add diagram for evaluation pipeline.

Evaluation pipeline is offline, using recorded MCAPs and metrics extraction.

The VictoriaMetrics + Grafana stack has been retired to a separate branch; analysis now uses the Parquet + plotting
scripts above.



## 📊 Experiment Grid

| Axis        | Values              | Notes                          |
|-------------|---------------------|--------------------------------|
| Platform    | RTX 3090, Orin NX   | Same ROS2 graph                |
| Precision   | FP32, FP16          | ONNX models                    |
| Threads     | 1, 2, 4             | `component_container` / `_mt`  |
| Replay rate | 1.0, 2.0, 3.0       | Multiplier on 10 Hz sequences  |
| Sequences   | All 21 KITTI train  | One run per sequence           |

→ **18 configurations × 2 platforms = 36 configs**, 21 sequences each.


### ⏱️ Logged KPIs

| Category              | Metrics                                      |
|-----------------------|----------------------------------------------|
| End-to-end latency    | Image stamp → track publish (p50/p95/p99)   |
| Detector              | Inference / callback latency, detections/frame, confidence stats |
| Tracker               | Callback duration, association latency |
| Systems               | Drop rate under `keep_last(1)`, CPU/GPU load |
| Ablations             | Executor threads, replay rate / load, (optional) FP16 on Jetson |

**Tracking quality** (TrackEval)
- HOTA, IDF1, MOTP (and the full TrackEval suite)


## Assets

TODO: create zip of the assets

- `results/results.parquet`,
- `models/yolo`,

Results are aggregated into a long-form Parquet file (≈20 MB+) published as a
[GitHub Release asset](https://github.com/jacobnzw/ros2-mot-pipeline/releases/tag/results_data).



## 🚀 Quick Start

1. Build the ROS2 workspace and source it.
2. Place KITTI MCAPs under the expected data path (see launch arguments).
3. Run a single configuration:
   ```bash
   ros2 launch launch/detector_test.launch.py \
     precision:=fp16 n_threads:=2 rate:=1.0 seq:=0019 config:=my_run
   ```
4. Or launch the full grid via `scripts/run_grid.py` / `scripts/eval/run_kitti_sequences.py`.
5. Build the analysis table and plots:
   ```bash
   uv run scripts/build_parquet.py
   uv run scripts/plot_results.py
   ```

Exact paths, model locations and dependency setup are documented in `docs/`.



## 📁 Repository Layout

```text
├── README.md
├── docs/                       # Protocol, architecture, detailed report
├── launch/                     # detector_test.launch.py
├── scripts/
│   ├── eval/                   # run_kitti_sequences.py, …
│   ├── run_grid.py
│   ├── build_parquet.py
│   └── plot_results.py
├── src/object_tracker/         # Detector + Tracker components
├── models/yolo/                # yolo11n_fp32.onnx, yolo11n_fp16.onnx, …
├── runs/                       # Per-config outputs (local)
└── data/                       # KITTI mcaps / labels (not committed)
```



*Built to understand the real cost of moving a ROS2 MOT pipeline from desktop to Jetson.*
