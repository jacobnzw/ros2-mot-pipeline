
# Terminology
| Term | Meaning |
|---|---|
| Trajectory | Actual path the object travels in the scene. |
| Track | Belief about object's trajectory. |
| Tracklet | A tracklet is a short, continuous segment of a detected object's trajectory across a few consecutive frames. Oclusion splits object's full trajectory into several separate tracklets. |


# Data: KITTI Tracking

## Preprocessing
**Letterboxing** is a preprocessing step that resizes an image **while preserving its aspect ratio**, then pads the remaining space (top/bottom or left/right) with a neutral fill value to reach the model's fixed input dimensions.

**Why it's needed:** Most models (e.g. YOLO) expect a fixed square input like 640×640. A naive `cv2.resize()` to that size would **distort** the image — stretching or squishing objects — which degrades detection accuracy.

**How it works:**

1. Compute the scale factor: `scale = min(target_h / src_h, target_w / src_w)`
2. Resize the image by that factor (aspect ratio preserved)
3. Create a canvas of the target size filled with a padding value
4. Paste the resized image centered on the canvas

**The padding value** is typically a neutral gray — **114** (used by YOLOv8/Ultralytics) or **127** — chosen so that after normalization (e.g. dividing by 255 and scaling to [-1, 1]) the padded pixels map to **zero activation**, meaning they carry no semantic information the model would mistake for real content.
A padded pixel at 0 contributes nothing to the dot products in the first conv layer, so the model effectively ignores it.
```
x_normalized = (x / 255.0 - 0.5) * 2.0   
```

**Trade-off vs. simple resize:**

| | Letterbox | Simple Resize |
|---|---|---|
| Aspect ratio | Preserved | Distorted |
| Compute cost | Slightly higher (padding is "wasted" pixels) | All pixels are useful |
| Object geometry | Faithful | Skewed |

The small accuracy gain from letterboxing is usually negligible for most applications, but it's the standard in object detection pipelines (YOLO, SSD, etc.) because it avoids the geometric distortion that can shift bounding box predictions.



## Object types
| ID | Type |
|----|------|
| 0 | `Car` |
| 1 | `Van` |
| 2 | `Truck` |
| 3 | `Pedestrian` |
| 4 | `Person_sitting` |
| 5 | `Cyclist` |
| 6 | `Tram` |
| 7 | `Misc` |
| 8 | `DontCare` |

## 
| Ultralytics (`kitti.yaml`) | TrackEval / KITTI label file |
|---|---|
| `car` | `Car` |
| `person` | `Pedestrian` |
| `truck` | `Truck` |


# ByteTrack
Available via `ultralytics` directly
```python
from ultralytics import YOLO
from ultralytics.tracker import TRACKERS

# Uses default bytetrack thresholds
model = YOLO("yolo11n.pt")
# YOLO detections w/ conf >= 0.1 passed to ByteTrack to handle the thresholding
result = model.track(frame, conf=0.1, tracker="bytetrack.yaml")  


# For more control over ByteTrack thresholds
bytetrack_args = TRACKERS['bytetrack']
bytetrack_args['track_high_thresh'] = 0.6   # Raise for cleaner tracks
bytetrack_args['new_track_thresh'] = 0.5    # Lower to prevent unassigned tracks
bytetrack_args['track_buffer'] = 60         # Keep lost targets in memory longer (60 frames)

results = model.track(frame, tracker=bytetrack_args)

# Works directly on videos too
model.track(source="video.mp4", tracker=bytetrack_args)
```

Uses
```
A simple Kalman filter for tracking bounding boxes *in image space*.

The 8-dimensional state space

    x, y, a, h, vx, vy, va, vh

contains the bounding box center position (x, y), aspect ratio a, height h,
and their respective velocities.

Object motion follows a constant velocity model. The bounding box location
(x, y, a, h) is taken as direct observation of the state space (linear
observation model).
```

# Multi-Object Tracking Metrics

## CLEAR Metrics
The two core metrics are
| Metric | What it measures | Formula |
|--------|-----------------|---------|
| **MOTA** | Overall accuracy (detection + association) | $1 - \frac{FN + FP + IDSW}{GT}$ |
| **MOTP** | Localization precision only | $\frac{\sum IoU_{matched}}{TP}$ |

The CLEAR family in `trackeval` also reports:
| Sub-metric | Meaning |
|---|---|
| **MT** (Mostly Tracked) | # GT tracks tracked >80% of their lifetime |
| **PT** (Partially Tracked) | # GT tracks tracked 20–80% |
| **ML** (Mostly Lost) | # GT tracks tracked <20% |
| **Frag** (Fragmentations) | Times a tracked object becomes untracked then re-tracked |
| **IDSW** (ID Switches) | Identity changes between consecutive frames |
| **FP / FN** | Total false positives / false negatives |
| **CLR_Recall / CLR_Precision** | Fraction of GT detected / fraction of detections that are correct |
| **MTR / PTR / MLR** | MT, PT, ML expressed as ratios of total GT tracks |
| **FAR** (False Alarm Rate) | FP per frame |
| **MODA** | $1 - FN/GT$ (accuracy without ID switches) |
| **sMOTA / MOTAL** | Variants that exclude or re-weight certain error types |

## HOTA (Higher Order Tracking Accuracy)
is widely considered the superior and more balanced metric for evaluating Multi-Object Tracking (MOT) compared to **MOTA (Multi-Object Tracking Accuracy)** because it equally weights detection and association accuracy. While **MOTA** is heavily biased toward detection performance (explaining over 99% of its score variance based on detection alone) and only measures "first-order" association (identity switches between consecutive frames), **HOTA** measures "higher-order" association across the entire video sequence.

Key differences include:

*   **Balanced Scoring**: **HOTA** decomposes into **DetA** (Detection Accuracy) and **AssA** (Association Accuracy), combining them via the geometric mean: $$HOTA_{\alpha} = \sqrt{DetA_{\alpha} \cdot AssA_{\alpha}}.$$ This prevents a tracker from achieving a high score by merely detecting objects well while failing to maintain consistent IDs.
*   **Localization**: **HOTA** incorporates localization accuracy directly into its sub-metrics, whereas **MOTA** requires a separate metric (**MOTP**) to measure localization, making **HOTA** a more comprehensive single-score evaluation.
*   **Human Alignment**: Studies show **HOTA** aligns significantly better with human visual assessment (agreeing with human evaluators ~61.6% of the time vs. 38.4% for **MOTA**), making it the current standard for benchmarking state-of-the-art tracking algorithms.



# Latency Measurement
```python
start_event = torch.cuda.Event(enable_timing=True)
start_event.record()
```
For `start_event.record()` queues a marker on the current CUDA stream; it doesn’t start a CPU stopwatch. 
If Python/Ultralytics takes time before queuing GPU work, the stream can sit idle after that marker. 
That idle gap can therefore appear in the event interval. The events measure stream elapsed time, not just the sum of 
GPU kernels. If your goal is kernel-only time, use a profiler; if your goal is how long model(input) takes to return 
with GPU work complete, your synchronized `perf_counter` timing is the clearer metric.

`DataLoader` gives you a CPU tensor, and Ultralytics’ prediction path moves tensor inputs to the model’s device. 
Leaving that transfer inside `model(input)` means your wall-clock measurement includes it. 
Move it to CUDA yourself only if you deliberately want to exclude host-to-device transfer, and do so before starting the timer.

Hence wall-clock CPU time is preferred using `time.perf_counter()`, this is per-Frame inference-call latency
```python
# ...

torch.cuda.synchronize()  # Wait for the GPU to finish any current work
start_time = time.perf_counter()

_ = model(input, verbose=False)  # GPU works ...

torch.cuda.synchronize()  # Wait for the GPU to finish
latency_ms = (time.perf_counter() - start_time) * 1000

# ...
```

On CUDA, the synchronized `perf_counter()` timing waits for the call’s GPU work to finish, so it captures more than just GPU kernel time: it includes CPU work inside `model(...)` and any input transfer Ultralytics performs there. It excludes KITTI loading and resizing, which happen before the timed call.

**For a meaningful Jetson comparison**, run the same script and metric on both devices with 
- the same model weights, 
- KITTI frames, 
- resolution, 
- precision, and 
- warmup count. 

Keep the Jetson power mode and thermal conditions consistent, and label the result “inference-call latency,” not “GPU-only latency.” Your per-frame median and p95 are useful; repeat the benchmark in multiple runs to see how much the results vary.


## Throuput vs Latency
| Metric | Definition | Favored by |
|--------|-----------|------------|
| **Latency** | Time from request arrival → response (per-request) | Small batch (1), fewer concurrent requests |
| **Throughput** | Total requests/tokens processed per second | Large batch, many concurrent requests |

The core mechanism: A GPU's parallelism only helps if you feed it parallel work. Batch size is the primary lever:

- **Batch size 1** → lowest per-request latency, but GPU is underutilized → low throughput
- **Large batch size** → GPU is saturated, throughput maximized, but each request waits for the batch to complete → higher per-request latency 

The relationship is **nonlinear**: going from batch 1→8 can give ~8× throughput with ~2× latency penalty, but 8→32 might only give 2× more throughput with 5× more latency (diminishing returns as you hit the compute-bound regime). 


# ONNX & TensorRT

> Always export to ONNX first. It's the de facto standard intermediate, and NVIDIA's own docs frame it as the primary path: "You typically start from a trained model exported to ONNX."

- **It's your debug checkpoint.** Validate the ONNX with onnxruntime on CPU before committing to a 30-min TensorRT build. Catches export bugs (shape mismatches, unsupported ops, custom op leaks) cheaply.
- **It's portable.** One .onnx file can feed TensorRT, ONNX Runtime on AMD/Intel/ARM, CoreML, TFLite, etc. The .engine is a dead end locked to one GPU + one TensorRT version.
- **It's versionable.** You can commit the ONNX to git, CI it, A/B test it, hand it to a teammate. An .engine is an opaque binary tied to your exact hardware.
- **TensorRT's primary input IS the ONNX parser.** Even when people say "go straight to TensorRT," they usually mean "skip ONNX Runtime" — not "skip the ONNX file." The OnnxParser is the standard ingestion path.

- **Build for latency**: set `max_batch_size`=1, tune `opt_shape` for single inference. Use `trtexec --batch=1` to benchmark.
- **Build for throughput**: set a larger `max_batch_size`, tune `opt_shape` for that batch. Throughput scales roughly linearly until you saturate compute.

You can build multiple engines (one per batch size) and dispatch at runtime based on current queue depth.


## Practical recommendation
```
PyTorch model
    │
    ▼
  .onnx  ← commit this, validate with onnxruntime, ship to team
    │
    ├──► ONNX Runtime (CPU / AMD / ARM / fallback)
    │
    └──► trtexec / TensorRT Builder → .engine (NVIDIA GPU, max perf)   
```
