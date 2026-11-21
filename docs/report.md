# Report

### Primary goal

> **Quantify the latency distribution and tracking-quality impact of moving a production-style ROS2 multi-object
> tracking pipeline from a high-end PC (RTX 3090) to a Jetson, under controlled MCAP replay of a real tracking
> sequence.**


### Report structure that falls out of this goal

1. **Introduction & goal**  
   Exact statement of the primary question + architectural choices (components, zero-copy, keep_last(1), offline
   observability).

2. **System under test**  
   Pipeline diagram, message types, QoS, executor configuration, measurement method (steady clocks, what is logged).

3. **Experimental protocol**  
   Sequence (KITTI 0019), replay method, platforms (RTX 3090 vs Jetson model), runs performed (baseline + threading +
   rate + optional precision).

4. **Latency results**  
   End-to-end and per-component distributions/percentiles on both platforms. Heatmaps or time-series if useful. Drop
   rates. Durations: detector/tracker callback; Latencies: model inference, LAS track association

5. **Tracking quality results**  
   ID switches / fragmentation (and MOTA if available) and any correlation with latency.

6. **Controlled ablations**  
   Threading effect, rate/load effect, (optional) precision effect on Jetson.

7. **Discussion**  
   Where time is spent, what breaks first on Jetson, whether the ROS2 design choices held up, practical implications for
   deployment.

8. **Conclusion & recommendation**  
   Clear statement: the pipeline is / is not viable on this Jetson class under the tested conditions, and what the next
   concrete step would be.



### What to measure (primary + high-value secondary)

**Core metrics (must have — already instrumented)**

| Category | Metric | Why it matters for the report |
|----------|--------|-------------------------------|
| End-to-end | Image stamp → track publish latency (p50 / p95 / p99, histogram) | The number that decides whether the pipeline is usable |
| Component | Detector callback / inference latency | Isolates the heavy stage |
| Component | Tracker callback duration | Shows whether association stays cheap |
| Systems | Drop / skip rate under keep_last(1) | Reveals whether the pipeline is losing data under load |
| Tracking KPI | ID switches, track fragmentation, (optional MOTA/IDF1 if easy) | Shows whether latency changes actually hurt tracking quality |

**High-value secondary measurements (low extra effort, high report value)**

1. **Executor threading**  
   Single-threaded vs MultiThreaded (threads_num = 2 or 4).  
   Same replay, same sequence.  
   → Directly answers “does the middleware concurrency model help or hurt on each platform?”

2. **Load / rate stress**  
   Replay at 1.0× vs 1.5× or 2.0× (or force higher input rate).  
   → Shows how the keep_last(1) design behaves when the pipeline is saturated. Especially interesting on Jetson.

3. **Platform resource snapshot**  
   CPU / GPU utilisation, memory, and (on Jetson) thermal / power if the rental exposes it.  
   → Turns the latency numbers into a systems story.

4. **Optional but hot: precision / backend on Jetson only**  
   Keep PC at full-precision ORT. On Jetson run the same ONNX in FP16 (and INT8 if it is quick).  
   → This is the single highest-leverage extra experiment for a Jetson report and is still modest scope.

Out of scope: new models, live cameras, TensorRT engine builds, multi-sequence leaderboards


### Alternative / hotter experiment ideas (still doable)

If you want a slightly different framing that still uses the same codebase:

- **“Can keep_last(1) + zero-copy ROS2 survive realistic load on Jetson?”**  
  Primary axis becomes drop rate and latency tail under increasing replay rate, with PC as the reference. Tracking KPIs
  become secondary.

- **“What does the detector latency distribution do to track stability?”**  
  Correlate per-frame detector latency (or jitter) with subsequent ID switches / fragmentation. Works on both platforms;
  Jetson just gives a more stressed distribution.

- **Precision ablation on the target platform**  
  FP32 vs FP16 (vs INT8) on Jetson only, same ROS2 graph. PC stays the reference full-precision run. This is the classic
  embedded story and maps cleanly onto a report section.

I would **not** replace the PC-vs-Jetson comparison; it is the natural narrative given that you are renting the board.
The alternatives above are best treated as focused secondary questions or report sections.


### Practical recommendation

Stick with the tightened primary goal above.  
Instrument (or confirm you already capture) the core metrics + the two easiest secondaries (threading and rate stress).  
Add FP16 on Jetson only if it costs less than a day.

That set is hot enough for a strong report, fully supported by the observability you already built, and small enough
that the rented Jetson time stays focused on measurement rather than new feature work.


TODO: Changing the QoS queue depth in .keep_last() in the detections handoff to the tracker had no effect on frame drop
counts


# Experiment Design & Protocol

## Platforms

### PC
The term "PC" refers to the following platform spec:

| Specification | Details |
|:--------------|:--------|
| Motherboard | ASUS ROG STRIX B650E-I GAMING WIFI |
| RAM | 64GB DDR5 |
| CPU | AMD Ryzen 9 7950X3D |
| GPU | MSI RTX 3090 Gaming X Trio, 24GB VRAM |
| Environment | WSL2 Ubuntu 24.04 |


### Jetson
The term "Jetson" refers to the following platform spec:

| Specification | Details |
|:--------------|:--------|
| Provider | [cloudjetson.com](https://cloudjetson.com) |
| Board | [Forecr DSBOARD-ORNX](https://www.forecr.io/products/carrier-board-dsboard-ornx) |
| SoM | [Orin NX 16GB](https://developer.download.nvidia.com/assets/embedded/secure/jetson/orin_nx/docs/Jetson_Orin_NX_DS-10712-001_v0.5.pdf?t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9&__token__=exp=1795098195~hmac=6608dc0c3a6e12b451eff756aa386069c42e3a6587d145723ce2caac2e037f4e) |
| Power mode | `MAXN_SUPER` |

| Host Environment | Version |
|:--------------|:--------|
| JetPack | 7.2.1 |
| Jetson Linux | 39.2.1 |
| CUDA | 13.2 |
| TensorRT | 10.16.2.10 |
| cuDNN | 9.20.0.46 |

The runtime environment was a Docker container built via
[`jetson-containers`](https://github.com/dusty-nv/jetson-containers/tree/master) with
```shell
CUDA_VERSION=13.2 jetson-containers build \
  --name=ros2-mot-jazzy \
  ros:jazzy-ros-base onnxruntime:1.25.1 \
  --skip-tests=all
```


## Run Configurations

I ran (2 x 3 x 3 = 18) design configurations on both platforms using the `scripts/run_grid.py` defined as Cartesian
product of the following axes:

```python
PRECISIONS = ("fp32", "fp16")
THREAD_COUNTS = (1, 2, 4)
RATES = (1.0, 2.0, 3.0)
```

## Measured KPIs

All logging was done via ReRun into an *.rrd files for easy viewing.

### Hardware

Hardware resource usage and sensor readings were measured to contextualize pipeline performance:

| Resource / sensor | Notes |
|:------------------|:------|
| CPU utilization | Utilization during replay |
| GPU utilization | Utilization during replay |
| Memory usage | System and GPU memory consumption, where available |
| Temperature | CPU/GPU or board temperatures reported by platform sensors |
| Power draw | System or board power reported by platform sensors |

Sensor availability and measurement scope may differ between the PC and Jetson. On PC, I relied on `psutils` and on
Jetson I used [`jetson-stats`](https://github.com/rbonghi/jetson_stats).

### ROS2 Pipeline

I measured and logged the following pipeline KPIs

| KPI | Notes |
|:----|:------|
| `pipeline/end_to_end_latency_ms` | Time from image timestamp to publishing the corresponding tracks; measures end-to-end responsiveness. |
| `detector/callback_interval_ms`  | Time between consecutive detector callback starts; indicates the effective processing cadence. |
| `detector/callback_duration_ms`  | Elapsed time for the detector callback, including preprocessing and inference. |
| `detector/preprocessing_latency_ms`  | Time spent preparing the image input for the model. |
| `detector/inference_latency_ms`  | Time spent running model inference. |
| `detector/frame_drop_count`  | Cumulative number of input frames dropped or skipped by the detector. |
| `tracker/callback_interval_ms` | Time between consecutive tracker callback starts; indicates the effective tracking cadence. |
| `tracker/callback_duration_ms` | Elapsed time for the tracker callback. |
| `tracker/association_latency_ms` | Time spent associating detections with existing tracks. |
| `tracker/active_tracks` | Number of tracks currently active during processing. |
| `tracker/frame_drop_count` | Cumulative number of frames dropped or skipped by the tracker. |


### Multi-Object Tracking

The resulting 2D tracks were also logged to compute the following MOT KPIs using
[`TrackEval`](https://github.com/JonathonLuiten/TrackEval) utilty:

|KPI | Notes |
|:---|:------|
| HOTA | Higher-Order Tracking Accuracy |
| IDF1 | ID switch F1 score |
| MOTP | Multi-Object Tracking Precision |


### Questions 
Given the platforms, the run configurations and the measured KPIs, the following questions about the ROS2 MOT pipeline
performance were analysed:

- **Platform Effect:** How do end-to-end latency, component latencies, frame drops, and MOT metrics change when moving
  from PC to Jetson at matched precision (FP32 and FP16 separately)?
- **Precision Effect:** What is the latency / frame drop / tracking-quality trade-off of FP16 vs FP32?
- **Executor Threading Effect:** Does increasing from 1 → 2 → 4 threads change latency tails, drop counts, or MOT
  metrics on either platform?
- **Load Effect:**
  - How do latency distributions and frame-drop counts scale with rate (1.0 / 2.0 / 3.0)?
  - At what point does the pipeline saturate on each platform?
- **Load x Platform Interaction:** Does the Jetson degrade more sharply than the PC as rate increases?
- **Deployment Effect:**
  - What's the MOT performance delta against PC @ FP32 baseline when I move the MOT pipeline to Jetson @ FP16 precision?
- **Resource Consumption:** What's the CPU / GPU utilisation, memory, and power / temperature under the different
  configurations?

Each of these questions are further elaborated on in the Results section.




# Results

TODO: figure out the type of plots to use for each effect


Aspects:
- Latency:
   - Metrics: e2e_{p50,p95,p99} [ms]
- Frame drops:
   - Metrics: total frames dropped [count]
- MOT:
   - Metrics: HOTA, IDF1, MOTP, IDSW, Frag [%]


## Platform Effect
Fix: rate=1.0, threads=2; Switch: *precision*

- [ ] bar chart per aspect: bar group by metric, bar per *platform*
- [ ] commentary

## Precision Effect
Fix: rate=1.0, threads=2; Switch: *platform*

- [ ] bar chart per aspect: bar group by metric, bar per *precision*;
- [ ] commentary

## Executor Threading Effect
Directly answers “does the middleware concurrency model help or hurt on each platform?”

- [x] Latency vs. thread count: Plot p50, p95, and p99 end-to-end latency for 1/2/4 threads; facet by platform and keep
  replay rate fixed.
- [x] Total frames Dropped vs. thread count: Plot dropped-frame percentage for the same configurations.
- [x] MOT KPI bar chart: Show HOTA, IDF1, and MOTP by thread count to check whether threading changes tracking quality.

![E2E Pipeline Latency: Jetson FP32 R1.0](../results/figures/executor_effect_jetson_fp32_r1.png)
![E2E Pipeline Latency: Jetson FP32 R1.0](../results/figures/executor_effect_pc_fp32_r1.png)


## Load Effect

Fix: precision=FP32 # higher latency model Switch: *platform*, *sequences="all"* # could select a particular sequence
Parameter: threads

Total frames dropped = sum of dropped frames across all sequences TODO: Line plot: X rate, Y total dropped frames; one
line per thread count


## Load x Platform Interaction

Why would there be an expected change in detector latency w/ increased rate? Doesn't make sense. Given some QoS, rates
affect number of processed frames, not per-frame latency.

TODO: fix n_threads=2, compare across rates inference latency histograms (tails)


## Deployment Effect

TODO: plots: PC FP32 vs Jetson FP16 w/ n_threads=2, rate=1.0: compare MOT KPIs and inference/association latencies


## Resource Consumption

TODO: summarise CPU/GPU usage, memory, and power for different configs