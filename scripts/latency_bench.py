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


def _model_latency(model: YOLO, input: torch.Tensor, tracker: bool, device) -> float:
    infer = model.track if tracker else model.predict

    if device.type == "cuda":
        torch.cuda.synchronize()  # Wait for the GPU to finish any current work

    start_time = time.perf_counter()
    # FIXME: missing tracker="bytetrack.yaml"
    infer(input, verbose=False)

    if device.type == "cuda":
        torch.cuda.synchronize()  # Wait for the GPU to finish

    return (time.perf_counter() - start_time) * 1000


def benchmark(args):

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
        _ = model.predict(dummy_input, verbose=False)

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
        # Input tensor `img` transfered to device by ultralytics API
        latency_ms = _model_latency(model, img, args.tracker, device)
        latencies.append(latency_ms)

    # Calculate Latency Percentiles
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)

    # Plot latency distribution.
    sns.histplot(latencies, bins="auto", kde=True)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    measured_entity = "YOLO+ByteTrack" if args.tracker else "YOLO"
    plt.title(f"{measured_entity} Inference-Call Latency Distribution")
    plt.tight_layout()
    histogram_path = f"results/figures/{measured_entity.lower()}_latency_histogram_baseline.png"
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
    print(f"\n=== {measured_entity} Benchmark Results ===")
    print(f"Device:         {device.type.upper()}")
    print(f"Input size:     {new_hw}")
    print(f"Median Latency: {median_latency:.2f} ms")
    print(f"P95 Latency:    {p95_latency:.2f} ms")
    if device.type == "cuda":
        print(f"{mem_type}: {peak_mem_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference-call latency of YOLO + ByteTrack.")
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
    parser.add_argument(
        "--tracker",
        action="store_true",
        default=False,
        help="Latency of detector + tracker is measured. Default: only detector latency measured.",
    )
    args = parser.parse_args()

    # Run the benchmark
    benchmark(args)
