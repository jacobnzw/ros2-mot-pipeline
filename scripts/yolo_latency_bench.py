import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

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
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

        # start_event.record()
        torch.cuda.synchronize()  # Wait for the GPU to finish any current work
        start_time = time.perf_counter()

        _ = model(input, verbose=False)  # GPU works ...

        torch.cuda.synchronize()  # Wait for the GPU to finish
        latency_ms = (time.perf_counter() - start_time) * 1000
        # end_event.record()
        # latency_ms = start_event.elapsed_time(end_event)  # Returns milliseconds
    else:
        # CPU timing
        start_time = time.perf_counter()
        _ = model(input, verbose=False)
        latency_ms = (time.perf_counter() - start_time) * 1000  # Convert to ms

    return latency_ms


def benchmark_yolo(args):

    torch.manual_seed(args.seed)

    # 1. Initialize device and model
    device = torch.device(args.device)
    # auto-downloaded to model_path if not already present
    model = YOLO(args.model_path).to(device)

    # Reset memory stats before starting
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup runs (ignores initial model loading overhead)
    new_hw = (
        math.ceil(args.scale * (KittiImages.HEIGHT // 32)) * 32,
        math.ceil(args.scale * (KittiImages.WIDTH // 32)) * 32,
    )
    dummy_input = torch.randn(1, 3, *new_hw).to(device)
    dummy_input /= dummy_input.max()
    print("Running warmup...")
    for _ in range(args.n_warmup_runs):
        _ = model(dummy_input, verbose=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    transform = transforms.Compose(
        [
            transforms.Resize(new_hw),
            transforms.ToTensor(),
        ]
    )
    dataset = KittiImages(args.kitti_seq, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    latencies = []
    for img in loader:
        # TODO: shouldn't img be transfered to GPU??
        latency_ms = _model_latency(model, img, device)
        latencies.append(latency_ms)

    # Calculate Latency Percentiles
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)

    # Plot latency distribution.
    sns.histplot(latencies, bins="auto", kde=True)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("YOLO Inference-Call Latency Distribution")
    plt.tight_layout()
    histogram_path = "results/figures/yolo_latency_histogram_baseline.png"
    plt.savefig(histogram_path)
    plt.close()
    print(f"Latency histogram saved: {histogram_path}")

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
    print(f"Input size:     {new_hw}")
    print(f"Median Latency: {median_latency:.2f} ms")
    print(f"P95 Latency:    {p95_latency:.2f} ms")
    if device.type == "cuda":
        print(f"{mem_type}: {peak_mem_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference-call latency of YOLO detector.")
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
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--n-warmup-runs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Run the benchmark
    benchmark_yolo(args)
