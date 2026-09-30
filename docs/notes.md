
Terminology
| Term | Meaning |
|---|---|
| Trajectory | Actual path the object travels in the scene. |
| Track | Belief about object's trajectory. |
| Tracklet | A tracklet is a short, continuous segment of a detected object's trajectory across a few consecutive frames. Oclusion splits object's full trajectory into several separate tracklets. |



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
