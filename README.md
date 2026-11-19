# ROS2 Multi-Object Tracking Pipeline — PC vs Jetson

**Latency & quality characterization of a production-style YOLO + ByteTrack ROS2 pipeline**

This repository contains a modular ROS2 multi-object tracking (MOT) system and the experimental harness used to measure
how moving it from a high-end PC to a Jetson affects end-to-end latency, component timings, and tracking quality.



## 🎯 Project Goal

Quantify the **latency distribution** and **tracking-quality impact** of running the same ROS2 MOT pipeline on:

- High-end PC (RTX 3090 24GB)
- NVIDIA Jetson Orin NX 16GB (rented instance)

under controlled MCAP replay of a KITTI tracking sequences.

The work focuses on systems behaviour that matters for robotics deployment:
- End-to-end and per-component latency (p50 / p95 / p99)
- Detector & tracker KPIs
- Queue / drop behaviour under `keep_last(1)` QoS
- Effect of executor threading and input load
- Optional precision ablation on Jetson (FP16)



## ✨ Key Features

- **Modular ROS2 components**
  - YOLO detector (ONNX Runtime, full precision)
  - ByteTrack tracker
  - Zero-copy intra-process communication between nodes
- **Correct timing**
  - Sensor timestamp propagation
  - Steady-clock latency & callback duration measurement
- **Offline observability**
  - Rerun (`.rrd`) for images, detections and time-series
  - VictoriaMetrics + Grafana stack for latency histograms, percentiles and heatmaps
- **Configurable execution**
  - `keep_last(1)` QoS throughout
  - MultiThreaded executor (selectable via launch argument)
- **Reproducible experiments**
  - Looped mcap replay of KITTI tracking sequence 0019 (10 Hz)
  - Scripts for percentile / histogram extraction



## 🏗️ Architecture (simplified)

```text
mcap / bag replay
        │
        ▼
┌───────────────────┐
│  Detector node    │  YOLO-ORT (full precision)
│  (detections)     │
└─────────┬─────────┘
          │ zero-copy
          ▼
┌───────────────────┐
│  Tracker node     │  ByteTrack
│  (tracks)         │
└─────────┬─────────┘
          │
          ▼
   Logger / Rerun / metrics
```

All communication uses ROS2 with carefully chosen QoS. Measurements are taken with steady clocks and logged for offline
analysis (no live online dashboard required).



## 📊 What Was Measured

| Category              | Metrics                                      |
|-----------------------|----------------------------------------------|
| End-to-end latency    | Image stamp → track publish (p50/p95/p99)   |
| Detector              | Inference / callback latency, detections/frame, confidence stats |
| Tracker               | Callback duration, ID switches, fragmentation |
| Systems               | Drop rate under `keep_last(1)`, CPU/GPU load |
| Ablations             | Executor threads, replay rate / load, (optional) FP16 on Jetson |



## 🚀 Quick Start (high level)

1. Build the ROS2 workspace and source it.
2. Launch the component container with the desired executor settings.
3. Replay the KITTI 0019 mcap (or your own bag) at the chosen rate.
4. Collect Rerun recordings and metrics.
5. Use the provided scripts + Grafana stack to generate latency distributions and reports.

Detailed commands and configuration live in the `docs/` and `scripts/` directories (see repository layout below).



## 📁 Repository Layout

```text
├── README.md
├── docs/                  # Architecture notes, experiment protocol, results
├── configs/               # Node & QoS parameters
├── scripts/               # Replay helpers, metric extraction, analysis
├── src/
│   ├── detector/          # YOLO-ORT ROS2 component
│   ├── tracker/           # ByteTrack ROS2 component
│   └── ...
├── results/               # Tables, figures, example recordings
└── docker/                # VictoriaMetrics + Grafana stack (optional)
```



## 🖥️ Platforms

| Platform       | Role                          | Notes                          |
|----------------|-------------------------------|--------------------------------|
| PC (RTX 3090)  | Reference / development       | Full-precision ORT ~40 ms p95 |
| Jetson         | Target evaluation             | Rented instance (cloudjetson) |

The same ROS2 graph and measurement pipeline are used on both platforms so differences can be attributed to the hardware
(and optional precision changes).



## 📌 Status & Scope

- ✅ Component architecture, zero-copy, timing, offline observability
- ✅ PC baseline on KITTI 0019
- 🔄 Jetson measurements (in progress / planned)
- ❌ Live camera drivers, multi-camera, 3D lifting, online dashboards (explicitly out of scope for this phase)

This project prioritises **reproducible offline characterisation** over a full production deployment. The results are
intended to support a clear go/no-go decision for further embedded work and to document practical ROS2 middleware
behaviour under realistic load.



## 📄 Citation / Report

A short engineering report summarising the PC vs Jetson comparison, latency distributions, tracking KPIs and lessons
learned is (or will be) available in `docs/results.md`.



*Built for learning practical ROS2 systems behaviour and for making an evidence-based decision about Jetson deployment.*
