
Terminology
| Term | Meaning |
|---|---|
| Trajectory | Actual path the object travels in the scene. |
| Track | Belief about object's trajectory. |
| Tracklet | A tracklet is a short, continuous segment of a detected object's trajectory across a few consecutive frames. Oclusion splits object's full trajectory into several separate tracklets. |


# ByteTrack

A simple Kalman filter for tracking bounding boxes *in image space*.

The 8-dimensional state space
```
    x, y, a, h, vx, vy, va, vh
```
contains the bounding box center position `(x, y)`, aspect ratio `a`, height `h`,
and their respective velocities.

Object motion follows a constant velocity model. The bounding box location
`(x, y, a, h)` is taken as direct observation of the state space (linear
observation model).


## Inference Latency Measurement
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

Hence wall-clock CPU time is preferred using `time.perf_counter()`
```python
# ...

torch.cuda.synchronize()  # Wait for the GPU to finish any current work
start_time = time.perf_counter()

_ = model(input, verbose=False)  # GPU works ...

torch.cuda.synchronize()  # Wait for the GPU to finish
latency_ms = (time.perf_counter() - start_time) * 1000

# ...
```


## ONNX & TensorRT

> Always export to ONNX first. It's the de facto standard intermediate, and NVIDIA's own docs frame it as the primary path: "You typically start from a trained model exported to ONNX."

- **It's your debug checkpoint.** Validate the ONNX with onnxruntime on CPU before committing to a 30-min TensorRT build. Catches export bugs (shape mismatches, unsupported ops, custom op leaks) cheaply.
- **It's portable.** One .onnx file can feed TensorRT, ONNX Runtime on AMD/Intel/ARM, CoreML, TFLite, etc. The .engine is a dead end locked to one GPU + one TensorRT version.
- **It's versionable.** You can commit the ONNX to git, CI it, A/B test it, hand it to a teammate. An .engine is an opaque binary tied to your exact hardware.
- **TensorRT's primary input IS the ONNX parser.** Even when people say "go straight to TensorRT," they usually mean "skip ONNX Runtime" — not "skip the ONNX file." The OnnxParser is the standard ingestion path.


### Practical recommendation
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
