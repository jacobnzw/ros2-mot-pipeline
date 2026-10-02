# Project Planning

## Phases

| Phase | Focus | Weeks | Approx. hours | Outcome |
|---|---|---:|---:|---|
| 1A | Offline baseline + ONNX | 0–2 | 25–30 | Working detector+tracker, measured numbers, ONNX path |
| 1B | ROS2 middleware | 3–5 | 30–35 | Instrumented nodes, timestamps, queues, bag replay |
| 1C | Package & decide | 6 | 8–10 | Clean repo, report, Jetson go/no-go |
| 2 (optional) | Jetson | later | — | Only if Phase 1A numbers justify it |

## Phase 1: Concept Validation

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
    - End-to-end pipeline latency (p50 / p95 / p99)
    - queue behaviour.
- [ ] CPU/GPU utilisation and peak memory are documented.
- [ ] A written decision is made: proceed to Jetson, extend ROS2 work, or stop.
- [ ] Results are reported honestly, including negative or partial findings.


### Week 0: Setup, dataset lock, architecture (5–7 h)
🏆 Goal: Freeze scope before coding.

Results

- [x] Repository created with basic structure.
- [x] One-sentence hypothesis written.
- [x] Detector and tracker chosen.
- [x] Datasets selected and partially downloaded.
- [x] Architecture diagram drafted.
- [x] Establish HW baseline: 
  - [x] Run YOLO detector on 200 frames
  - [x] Record median/p95 latency + peak memory on this machine.
- [x] Environment reproducible


### Weeks 1–2: Offline baseline (20–25 h)

🏆 Goal: Produce measured detector → tracker pipeline (no ROS yet)

#### Results to achieve

- [ ] YOLO11-N + ByteTrack running on the chosen sequences.
- [ ] Annotated output video with track IDs.
- [ ] Basic detection and tracking metrics recorded.

#### Measurements

- [ ] Tracker latency (median, p95, p99) and throughput.
- [ ] Tracking metrics (MOTA/IDF1/HOTA or ID switches + fragmentation).
- [ ] Peak CPU/GPU memory and utilisation.

#### Risks

Main concern is getting ROS2 middleware experience.

IF we care about tracker correctness:
- ByteTrack doesn't ego-motion compensate: suitable for KITTI?


#### Exit / pivot

A valid result is a clean offline Python pipeline


### Weeks 3-5: ROS2 Middleware (30 - 35h)

🏆 Goal: Learn and demonstrate the middleware concepts that matter for robotics.

#### Results to achieve

- [ ] Python ROS2 nodes with clear interfaces.
- [ ] Bounded queues (depth 1–3).
- [ ] Correct timestamp propagation (sensor time, not wall time).
- [ ] Deterministic bag/file replay command.
- [ ] Per-stage latency logging (receive → inference start → publish).
- [ ] Basic diagnostics (queue depth, drop count).
- [ ] Simple visualisation of tracks.

#### Measurements

- [ ] End-to-end latency (t_track_output – t_image_stamp).
- [ ] Per-stage latencies (p50/p95/p99).
- [ ] Queue depth and drop rate under sustained replay.
- [ ] Topic rates (ros2 topic hz).


### Week 6: 

🏆 Goal: clean up repo, prep summary report, decision on Jetson

#### Results

- [ ] Repository cleaned; one-command offline benchmark and one-command ROS2 replay.
- [ ] Short engineering report.
- [ ] Interview-ready one-page summary.
- [ ] Explicit decision: proceed to Jetson, continue ROS2 work, or stop.


## Phase 2: Jetson Deployment

> Proceed to Jetson only if:
> - The offline PyTorch pipeline sustains roughly >15–20 FPS on your current hardware with usable tracking quality, and
> - The ROS2 nodes demonstrate correct timestamp propagation and acceptable queue behaviour.


- [ ] Model exported to ONNX and numerically + visually validated against PyTorch.
- [ ] Inference latency comparison: PyTorch vs ONNX Runtime (CPU + CUDA if available).
- [ ] Optional stretch: TensorRT engine on PC (document any failure; engine is not expected to be portable to Jetson).
