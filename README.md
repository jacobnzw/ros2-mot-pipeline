# Embedded-ready Multi-Object Tracking Pipeline for Robotics

Gaining experience with
- PyTorch/ONNX/TensorRT
- 👉 ROS2 middleware
- 🏆 Jetson deployment (🤞)


## Hypothesis
> “A minimal YOLOv8-N/YOLO11-N + ByteTrack pipeline can be instrumented as three clean ROS2 nodes with correct sensor-time stamp propagation and bounded queues, delivering measurable end-to-end latency, tracking quality, and resource numbers on a PC that will decide whether a later Jetson deployment is justified.”

```mermaid
flowchart LR
    A["Image source"] --> B["YOLO <br/> (PyTorch/ONNX)"]
    B --> C["ByteTrack <br/> (or OC-SORT fallback)"]
    C --> D["Tracked objects <br/> ID, box, class, (vel)"]
```

## Language choice

- Python (`rclpy`) for the entire Phase 1.
- C++ rewrite of the hot path is explicitly deferred (optional later exercise).

## TODO

- [x] Select and partially download the datasets
    - One annotated sequence for metrics: MOT17 or KITTI Tracking.
    - One ROS-compatible source: a public ROS bag or a short self-recorded sequence (plan the recording this week if none exists).

    👉 Download only what is needed for the first baseline eval runs. Note the exact sequences and any preprocessing steps in `docs/reproducibility.md`.
- [x] Lock the detector and tracker (YOLOv8-N / YOLO11-N + ByteTrack)
    - [ ] Record the exact model weights and library versions you will use.
- [ ] Create the experiment table
    - Detection latency (median / p95 / p99)
    - Throughput (FPS)
    - Tracking metrics (MOTA / IDF1 / HOTA or ID-switches + fragmentation)
    - Peak CPU / GPU memory and utilisation
    - ONNX file size and numerical difference vs PyTorch
    - End-to-end pipeline latency and queue behaviour (later)
- [ ] Run the hardware baseline (critical checkpoint)
    - Load the chosen YOLO model.
    - Run inference on ~200 frames from the selected sequence.
    - Record median and p95 latency + peak memory on the current machine.
    - Write the numbers down immediately.
    
    ➡️ Checkpoint: Environment must be reproducible and the hardware baseline numbers written down. If either is missing, stop and fix it before proceeding to Weeks 1-2