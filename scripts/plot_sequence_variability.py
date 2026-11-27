#!/usr/bin/env python3
"""Plot end-to-end latency and missing samples across repeated sequence runs."""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
LATENCY = "pipeline_end_to_end_latency_ms"
NANOSECONDS = 1_000_000_000


def variability_data(
    frame: pd.DataFrame, config: str, sequence: str, fps: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source FPS must be finite and greater than zero")
    if "ros2_timestamp_ns" not in frame:
        raise ValueError("Parquet lacks ros2_timestamp_ns; re-export recordings and rebuild the Parquet file")

    selected = frame[(frame["config"] == config) & (frame["sequence"] == sequence)]
    inventory = selected[(selected["source"] == "sequence") & (selected["metric"] == "input_frame_count")]
    if len(inventory) != 1:
        raise ValueError(f"{config}/{sequence}: expected exactly one sequence frame count")
    count = inventory["value"].iloc[0]
    if not math.isfinite(count) or count <= 0 or count != int(count):
        raise ValueError(f"{config}/{sequence}: frame count must be a positive integer")
    expected = [round(index * NANOSECONDS / fps) for index in range(int(count))]
    if len(set(expected)) != len(expected):
        raise ValueError("source FPS produces duplicate nanosecond timestamps")

    pipeline = selected[selected["source"] == "pipeline"]
    if pipeline.empty or pipeline["run"].isna().any() or pipeline["run"].eq("").any():
        raise ValueError(f"{config}/{sequence}: missing pipeline run IDs")
    run_count = pipeline["run"].nunique()
    latency = pipeline[pipeline["metric"] == LATENCY].copy()
    if latency.empty:
        raise ValueError(f"{config}/{sequence}: no end-to-end latency samples")
    if latency["ros2_timestamp_ns"].isna().any():
        raise ValueError(f"{config}/{sequence}: latency samples lack source timestamps; re-export recordings")
    if not pd.api.types.is_integer_dtype(latency["ros2_timestamp_ns"].dtype):
        raise ValueError("source timestamps must be integer nanoseconds")
    if not latency["value"].map(math.isfinite).all():
        raise ValueError(f"{config}/{sequence}: non-finite latency samples")
    if latency.duplicated(["run", "ros2_timestamp_ns"]).any():
        raise ValueError(f"{config}/{sequence}: duplicate source timestamp in a run")
    if not latency["ros2_timestamp_ns"].isin(expected).all():
        raise ValueError(f"{config}/{sequence}: source timestamps do not match frame count and FPS")

    frequencies = latency.groupby("ros2_timestamp_ns")["run"].nunique()
    counts = pd.DataFrame({"ros2_timestamp_ns": expected})
    counts["observed_runs"] = counts["ros2_timestamp_ns"].map(frequencies).fillna(0).astype(int)
    counts["missing_runs"] = run_count - counts["observed_runs"]
    counts["source_seconds"] = counts["ros2_timestamp_ns"] / NANOSECONDS
    latency["observed_runs"] = latency["ros2_timestamp_ns"].map(frequencies).astype(int)
    latency["source_seconds"] = latency["ros2_timestamp_ns"] / NANOSECONDS
    return latency, counts, run_count


def stacked_latency_matrix(latency: pd.DataFrame, counts: pd.DataFrame, run_count: int) -> np.ndarray:
    matrix = np.full((run_count, len(counts)), np.nan)
    frame_indices = {timestamp: index for index, timestamp in enumerate(counts["ros2_timestamp_ns"])}
    for timestamp, group in latency.groupby("ros2_timestamp_ns"):
        values = np.sort(group["value"].to_numpy(dtype=float))
        matrix[: len(values), frame_indices[timestamp]] = values
    return matrix


def plot_variability(
    frame: pd.DataFrame, config: str, sequence: str, fps: float, bin_seconds: float, output_dir: Path
) -> list[Path]:
    if not math.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("histogram bin width must be finite and greater than zero")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."} for part in (config, sequence)):
        raise ValueError("config and sequence must be single safe path components")
    latency, counts, run_count = variability_data(frame, config, sequence, fps)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    matrix = stacked_latency_matrix(latency, counts, run_count)
    edges = [0, 17, 20, 25, 30, 35, 40, 50, 75, 100, 250, 500, 1000, 1600]
    edges[0] = min(0.0, float(latency["value"].min()))
    edges[-1] = max(1600.0, float(latency["value"].max()))
    colors = plt.get_cmap("viridis", len(edges) - 1).with_extremes(bad="#e6e6e6")
    norm = BoundaryNorm(edges, colors.N, clip=True)
    figure, axes = plt.subplots(2, 1, figsize=(17, 7), layout="constrained")
    extent = (-0.5 / fps, (len(counts) - 0.5) / fps, 0.5, run_count + 0.5)
    for axis in axes:
        image = axis.imshow(
            matrix, origin="lower", aspect="auto", interpolation="nearest", extent=extent, cmap=colors, norm=norm
        )
        axis.set_ylabel("Sample rank")
        axis.set_xlabel("Source sequence time [s]")
        axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(run_count, 10)))
    axes[0].set_title("Full sequence")
    axes[0].set_xlim(extent[0], extent[1])
    detail_frames = max(1, math.ceil(min(10 * fps, len(counts))))
    detail_end = (detail_frames - 0.5) / fps
    axes[1].set_title(f"Detail: first {detail_frames / fps:g} seconds")
    axes[1].set_xlim(extent[0], detail_end)
    if detail_frames <= 200:
        axes[1].vlines(
            (np.arange(detail_frames - 1) + 0.5) / fps,
            0.5,
            run_count + 0.5,
            colors="white",
            alpha=0.25,
            linewidth=0.4,
        )
    axes[1].hlines(np.arange(1.5, run_count + 0.5), extent[0], detail_end, colors="white", alpha=0.25, linewidth=0.4)
    figure.suptitle(f"Sequence {sequence}: end-to-end latency ({run_count} runs)")
    # figure.supxlabel("Each column is a source frame; samples are sorted by latency. Gray cells = no sample.")
    figure.colorbar(image, ax=axes, boundaries=edges, ticks=edges, spacing="uniform", label="Latency [ms] (binned)")
    for extension in ("svg", "png"):
        path = output_dir / f"{config}_latency_variability.{extension}"
        figure.savefig(path, bbox_inches="tight", dpi=200)
        paths.append(path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4))
    end = counts["source_seconds"].iloc[-1]
    bins = np.arange(math.floor(end / bin_seconds) + 2) * bin_seconds
    axis.hist(counts["source_seconds"], bins=bins, weights=counts["missing_runs"], edgecolor="black")
    axis.set(
        title=f"Sequence {sequence}: missing end-to-end samples ({run_count} runs)",
        xlabel="Source sequence time [s]",
        ylabel=f"Missing samples per {bin_seconds:g} s",
    )
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    for extension in ("svg", "png"):
        path = output_dir / f"{config}_missing_samples.{extension}"
        figure.savefig(path, bbox_inches="tight", dpi=200)
        paths.append(path)
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Evaluated results Parquet containing source timestamps")
    parser.add_argument("--config", required=True, help="Configuration name")
    parser.add_argument("--sequence", required=True, help="Sequence ID")
    parser.add_argument("--fps", type=float, default=10.0, help="MCAP generation FPS, not playback rate (default: 10)")
    parser.add_argument("--bin-seconds", type=float, default=5.0, help="Missing-sample histogram bin width")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    try:
        frame = pd.read_parquet(args.parquet, dtype_backend="numpy_nullable")
        for path in plot_variability(frame, args.config, args.sequence, args.fps, args.bin_seconds, args.output_dir):
            print(f"Wrote {path}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
