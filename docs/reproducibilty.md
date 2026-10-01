# Reproducibility

| Component | Choice | Reasons |
|---| ---|---|
| **Dataset** | KITTI Tracking: 0007, 0019 | Complementary # cars and peds |
| **Detector** | YOLO11-N | Better detection accuracy for smaller or overlapping objects |
| **Tracker** | ByteTrack | Very fast: it's only associator + KF|

- One annotated sequence for metrics: KITTI Tracking.
- One ROS-compatible source: a public ROS bag or a short self-recorded sequence (plan the recording this week if none exists).

## Dataset
KITTI Multi-Object Tracking sequences

I chose the following eval sequences
- 0007
    - mostly cars, vans, trucks, few peds
- 0019
    - mostly peds (crowded area), few cars, longest

by Running analysis script

`uv run scripts/kitti_label_stats.py data/kitti/left_color/training/label_02`

and inspecting the table showing object type counts across frames. Choosing sequences that are somewhat complementary in the number of `Pedestrian` / `Person`/ `Cyclist` and `Car`/`Van`/`Truck`.

| Sequence | Length | Car | Cyclist | DontCare | Misc | Pedestrian | Person | Tram | Truck | Van | Objects Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0000 | 154 | 243 | 154 | 378 | 0 | 22 | 0 | 0 | 0 | 292 | 1089 |
| 0001 | 447 | 2681 | 0 | 1241 | 20 | 112 | 0 | 0 | 77 | 140 | 4271 |
| 0002 | 233 | 1032 | 75 | 601 | 16 | 180 | 0 | 0 | 84 | 110 | 2098 |
| 0003 | 144 | 363 | 0 | 473 | 0 | 0 | 0 | 0 | 0 | 25 | 861 |
| 0004 | 314 | 818 | 60 | 899 | 0 | 65 | 0 | 51 | 27 | 92 | 2012 |
| 0005 | 297 | 1275 | 139 | 672 | 0 | 0 | 0 | 0 | 30 | 32 | 2148 |
| 0006 | 270 | 550 | 0 | 684 | 0 | 0 | 0 | 0 | 101 | 111 | 1446 |
| 0007 | 800 | 2258 | 0 | 988 | 121 | 67 | 0 | 0 | 58 | 230 | 3722 |
| 0008 | 390 | 1046 | 0 | 717 | 2 | 0 | 0 | 0 | 30 | 293 | 2088 |
| 0009 | 803 | 2859 | 0 | 1525 | 0 | 29 | 0 | 0 | 612 | 276 | 5301 |
| 0010 | 294 | 603 | 14 | 395 | 59 | 30 | 0 | 127 | 25 | 70 | 1323 |
| 0011 | 373 | 3405 | 0 | 571 | 0 | 201 | 0 | 0 | 0 | 182 | 4359 |
| 0012 | 78 | 144 | 41 | 105 | 0 | 64 | 0 | 0 | 0 | 0 | 354 |
| **0013** | 340 | 55 | 237 | 935 | 18 | 929 | 167 | 0 | 0 | 69 | 2410 |
| 0014 | 106 | 455 | 0 | 149 | 0 | 122 | 0 | 0 | 0 | 72 | 798 |
| 0015 | 376 | 899 | 537 | 1282 | 25 | 752 | 0 | 0 | 0 | 0 | 3495 |
| 0016 | 209 | 836 | 272 | 1010 | 0 | 2027 | 0 | 0 | 0 | 0 | 4145 |
| 0017 | 145 | 0 | 101 | 616 | 0 | 782 | 0 | 0 | 0 | 0 | 1499 |
| 0018 | 339 | 1354 | 0 | 381 | 0 | 0 | 0 | 0 | 0 | 59 | 1794 |
| **0019** | 1059 | 927 | 308 | 2366 | 91 | 6088 | 509 | 417 | 0 | 486 | 11192 |
| 0020 | 837 | 5497 | 0 | 2051 | 441 | 0 | 0 | 0 | 145 | 762 | 8896 |

Class Remapping Table
| KITTI class | YOLO11-n (COCO) class(es) | Recommendation for Phase 1 |
|---|---|---|
| Car | car | Keep (direct) |
| Van | car (or truck) | Map to Car |
| Truck | truck | Keep or map to Car |
| Pedestrian | person | Keep (direct) |
| Person_sitting | person | Map to Pedestrian |
| Cyclist | person + bicycle | Usually treat as Pedestrian (or ignore) |
| Tram | train / bus | Optional / ignore |
| Misc / DontCare | — | Discard |


## PC Baseline
Reproduce with
```shell
uv run scripts/yolo_latency_bench.py --kitti-seq path/to/downloaded/kitti/left_color/training/image_02/0019
```
Numbers should be similar

Results

| Metric | Value |
|---|---|
| Device | nVidia RTX3090 24GB VRAM  |
| Dataset | KITTI Tracking 0019  |
| Input resolution | (352, 1216)  |
| Frames | 1059  |
| Median Latency | 12.91 ms |
| P95 Latency | 14.43 ms |
| GPU Peak Memory | 82.03 MB |

Using image resolution `(352, 1216)` as that's the closest multiple of `32` to the original `(375, 1242)`, required by the YOLO11-N detector.

<!-- ![Baseline Latency](../results/figures/yolo_latency_histogram_baseline.png) -->
<p align="center">
  <img src="../results/figures/yolo_latency_histogram_baseline.png" height="400">
</p>
