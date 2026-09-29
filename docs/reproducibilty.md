### Lock the detector and tracker

Detector: YOLOv8-N or YOLO11-N (start with the smaller nano variant).
Tracker: ByteTrack (hard timebox of 4 h; fall back to OC-SORT if integration stalls).

Record the exact model weights and library versions you will use.

### Select and partially download the datasets

One annotated sequence for metrics: MOT17 or KITTI Tracking.
One ROS-compatible source: a public ROS bag or a short self-recorded sequence (plan the recording this week if none exists).

Download only what is needed for the first baseline runs. Note the exact sequences and any preprocessing steps in docs/reproducibility.md.

### Create the experiment table
Make a simple table (spreadsheet or markdown) that will hold:

- Detection latency (median / p95 / p99)
- Throughput (FPS)
- Tracking metrics (MOTA / IDF1 / HOTA or ID-switches + fragmentation)
- Peak CPU / GPU memory and utilisation
- ONNX file size and numerical difference vs PyTorch
- End-to-end pipeline latency and queue behaviour (later)

### Run the hardware baseline (critical checkpoint)

- Load the chosen YOLO model.
- Run inference on ~200 frames from the selected sequence.
- Record median and p95 latency + peak memory on the current machine.
- Write the numbers down immediately.

Checkpoint: Environment must be reproducible and the hardware baseline numbers written down. If either is missing, stop and fix it before proceeding to Weeks 1-2