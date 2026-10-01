import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO
import seaborn as sns
import matplotlib.pyplot as plt

YOLOv11n = "models/yolo/yolo11n.pt"


class KittiImages(Dataset):
    WIDTH = 1238
    HEIGHT = 374

    def __init__(self, img_dir: str, ext: str = "png", transform=None):
        self.paths = sorted(Path(img_dir).glob(f"*.{ext}"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):  # ty: ignore[invalid-method-override]
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def _model_latency(model, input, device) -> float:
    if device.type == "cuda":
        # GPU precise timing using CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = model(input, verbose=False)
        end_event.record()

        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)  # Returns milliseconds
    else:
        # CPU timing
        start_time = time.perf_counter()
        _ = model(input, verbose=False)
        latency_ms = (time.perf_counter() - start_time) * 1000  # Convert to ms

    return latency_ms


def _synthetic_benchmark_loop(model, dummy_input, num_runs, img_size=640, device="cuda"):
    """Synthetic bench feeding dummy_input into the model."""

    # Latency benchmarking loop
    print(f"Benchmarking {num_runs} runs...")
    latencies = []

    for _ in range(num_runs):
        latency_ms = _model_latency(model, dummy_input, device)
        latencies.append(latency_ms)

    return latencies


def _kitti_benchmark_loop(model, img_dir, img_scale=1.0, device="cuda"):
    new_hw = (
        math.ceil(img_scale * (KittiImages.HEIGHT // 32)) * 32,
        math.ceil(img_scale * (KittiImages.WIDTH // 32)) * 32,
    )
    print(f"{new_hw=}")
    transform = transforms.Compose(
        [
            transforms.Resize(new_hw),
            transforms.ToTensor(),
        ]
    )
    dataset = KittiImages(img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    latencies = []
    for img in loader:
        latency_ms = _model_latency(model, img, device)
        latencies.append(latency_ms)

    return latencies


def benchmark_yolo(args):
    # 1. Initialize device and model
    device = torch.device(args.device)
    # auto-downloaded to model_path if not already present
    model = YOLO(args.model_path).to(device)

    # Reset memory stats before starting
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup runs (ignores initial model loading overhead)
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    dummy_input /= dummy_input.max()
    print("Running warmup...")
    for _ in range(args.n_warmup_runs):
        _ = model(dummy_input, verbose=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    if args.kitti_seq:
        latencies = _kitti_benchmark_loop(model, args.kitti_seq, args.kitti_scale, device)
    else:
        latencies = _synthetic_benchmark_loop(model, dummy_input, args.n_runs, args.img_size, device)

    # Calculate Latency Percentiles
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)

    # Plot latency distribution.
    sns.histplot(latencies, bins="auto", kde=True)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("YOLO Latency Distribution")
    plt.tight_layout()
    plt.savefig("results/figures/yolo_latency_histogram_baseline.png")
    plt.close()

    # Measure Peak Memory
    if device.type == "cuda":
        # Returns memory in bytes, convert to Megabytes
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        mem_type = "GPU Peak Memory"
    else:
        # Fallback if you want to implement psutil/tracemalloc for CPU memory
        peak_mem_mb = 0
        mem_type = "CPU Memory tracking requires separate profile"

    # Print Results
    print("\n=== Benchmark Results ===")
    print(f"Device:         {device.type.upper()}")
    # print(f"Image size:     {args.img_size}")
    print(f"Median Latency: {median_latency:.2f} ms")
    print(f"P95 Latency:    {p95_latency:.2f} ms")
    if device.type == "cuda":
        print(f"{mem_type}: {peak_mem_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark latency of YOLO detector.")
    # TODO: Subcommand for synthetic / kitti?
    parser.add_argument(
        "--model-path",
        default=YOLOv11n,
        # required=False,
        help="Path to YOLO model *.pt file",
    )
    parser.add_argument(
        "--kitti-seq",
        type=str,
        default="",
        required=False,
        help="Path to KITTI sequence.",
    )
    parser.add_argument("--kitti-scale", type=float, default=1.0)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-warmup-runs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Run the benchmark
    benchmark_yolo(args)
