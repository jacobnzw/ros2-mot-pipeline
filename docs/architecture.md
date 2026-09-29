# Architecture

```mermaid
flowchart LR
    A["Image source"] --> B["YOLO <br/> (PyTorch/ONNX)"]
    B --> C["ByteTrack <br/> (or OC-SORT fallback)"]
    C --> D["Tracked objects <br/> ID, box, class, (vel)"]
```

### Hypothesis
> “A minimal YOLOv8-N/YOLO11-N + ByteTrack pipeline can be instrumented as three clean ROS2 nodes with correct sensor-time stamp propagation and bounded queues, delivering measurable end-to-end latency, tracking quality, and resource numbers on a PC that will decide whether a later Jetson deployment is justified.”

### 🚧 Folder Structure

```
README.md
docs/
  architecture.md
  results.md
  reproducibility.md
configs/
scripts/
  run_offline_baseline.py
  ...
src/
  detector/
  tracker/
  ros2_nodes/
results/
  tables/
  figures/
```
