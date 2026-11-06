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