# Project Planning

## Phases

| Phase | Focus | Weeks | Approx. hours | Outcome |
|---|---|---:|---:|---|
| 1A | Offline baseline + ONNX | 0–2 | 25–30 | Working detector+tracker, measured numbers, ONNX path |
| 1B | ROS2 middleware | 3–5 | 30–35 | Instrumented nodes, timestamps, queues, bag replay |
| 1C | Package & decide | 6 | 8–10 | Clean repo, report, Jetson go/no-go |
| 2 (optional) | Jetson | later | — | Only if Phase 1A numbers justify it |

## Phase 1A

> “A minimal YOLOv8-N/YOLO11-N + ByteTrack pipeline can be instrumented as three clean ROS2 nodes with correct sensor-time stamp propagation and bounded queues, delivering measurable end-to-end latency, tracking quality, and resource numbers on a PC that will decide whether a later Jetson deployment is justified.”

| Category | Details |
|---|---|
| **Primary learning goal** | Gain practical ROS2 middleware experience (nodes, timestamps, bounded queues, bag replay, diagnostics) while building a measurable detection → tracking pipeline. |
| **Secondary goal** | Establish PC baselines (latency, accuracy, resource use) that inform a possible later Jetson deployment. |
| **Target duration** | 6 weeks (approximately 60–75 hours total). |
| **Scope rule** | Do not purchase a Jetson until Phase 1A produces clear latency and throughput numbers on the current machine. |

### Minimal deliverables

- [ ] Reproducible offline detector + tracker baseline with ONNX validation.
- [ ] Working ROS2 three-node pipeline (image source → detector → tracker) with timestamps and bounded queues.
- [ ] Latency, tracking-metric, and resource report on at least two sequences.
- [ ] Clear go/no-go recommendation for Jetson.
- [ ] Honest conclusion regardless of whether “real-time” targets are met.

### Success criteria (neutral)
Phase 1 is successful if and only if:

- [ ] A third party can run the offline benchmark and the ROS2 bag replay from the README.
- [ ] At least two sequences are evaluated with:
    - Detection latency (median / p95).
    - Tracking metrics (MOTA / IDF1 / HOTA or ID-switch + fragmentation counts).
    - End-to-end pipeline latency (p50 / p95 / p99) and queue behaviour.
- [ ] CPU/GPU utilisation and peak memory are documented.
- [ ] A written decision is made: proceed to Jetson, extend ROS2 work, or stop.
- [ ] Results are reported honestly, including negative or partial findings.


### Week 0: Setup, dataset lock, architecture (5–7 h)
🏆 Goal: Freeze scope before coding.

Results

- [x] Repository created with basic structure.
- [x] One-sentence hypothesis written.
- [ ] Detector and tracker chosen.
- [ ] Datasets selected and partially downloaded.
- [x] Architecture diagram drafted.
- [ ] Experiment table created.
- [ ] Hardware baseline: run the chosen YOLO model on 200 frames and record median/p95 latency + peak memory on this machine.

Dataset rule

- One annotated sequence for metrics (MOT17 or KITTI Tracking).
- One ROS-compatible source (public bag or short self-recorded sequence). Plan the recording in Week 0 if needed.

Checkpoint

- Environment reproducible and hardware baseline numbers written down. If not, stop and fix reproducibility first.


### Weeks 1–2: Offline baseline + ONNX (20–25 h)

🏆 Goal: Produce a correct, measured detector → tracker pipeline and a validated ONNX path.

#### Results to achieve

- [ ] YOLOv8-N/YOLO11-N running on the chosen sequences.
- [ ] ByteTrack integrated (switch to OC-SORT after 4 h max if integration fails).
- [ ] Annotated output video with track IDs.
- [ ] Basic detection and tracking metrics recorded.
- [ ] Model exported to ONNX and numerically + visually validated against PyTorch.
- [ ] Inference latency comparison: PyTorch vs ONNX Runtime (CPU + CUDA if available).
- [ ] Optional stretch: TensorRT engine on PC (document any failure; engine is not expected to be portable to Jetson).

#### Measurements

- [ ] Detection latency (median, p95, p99) and throughput.
- [ ] Tracking metrics (MOTA/IDF1/HOTA or ID switches + fragmentation).
- [ ] Peak CPU/GPU memory and utilisation.
- [ ] ONNX file size and numerical difference vs PyTorch.

#### Timeboxing

- Tracker debugging ≤ 4 h → switch.
- TensorRT debugging ≤ 6 h → fall back to ONNX Runtime and document.

#### Exit / pivot

A valid result is a clean offline Python pipeline + ONNX path even if TensorRT fails.