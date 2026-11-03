#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from statistics import quantiles

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import rerun as rr
import rerun_bindings
import seaborn as sns

SIGNALS = (
    ("/detector/inference_latency_ms", "Detector inference latency"),
    ("/detector/callback_interval_ms", "Detector callback interval"),
    ("/tracker/callback_interval_ms", "Tracker callback interval"),
    ("/pipeline/end_to_end_latency_ms", "Pipeline end-to-end latency"),
)
SCALAR_COMPONENT = "Scalars:scalars"


def read_latency_samples(recording_path: Path) -> dict[str, list[float]]:
    reader = rerun_bindings.RrdReaderInternal(str(recording_path))
    entity_paths = [entity_path for entity_path, _ in SIGNALS]
    stream = rr.chunk.LazyChunkStream(reader.stream()).filter(
        content=entity_paths,
        components=SCALAR_COMPONENT,
    )

    samples = {entity_path: [] for entity_path in entity_paths}
    for chunk in stream:
        values = chunk.to_record_batch().column(SCALAR_COMPONENT).to_pylist()
        for row_values in values:
            if row_values is None:
                continue
            samples[chunk.entity_path].extend(
                float(value) for value in row_values if value is not None and math.isfinite(value)
            )

    return samples


def save_histograms(samples: dict[str, list[float]], output_path: Path, recording_name: str) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
    figure.suptitle(f"Latency distributions - {recording_name}", fontsize=15)

    for axis, (entity_path, title) in zip(axes.flat, SIGNALS, strict=True):
        values = samples[entity_path]
        axis.set_title(title)
        axis.set_xlabel("Latency (ms)")
        axis.set_ylabel("Samples")
        if not values:
            axis.text(0.5, 0.5, "No samples found", ha="center", va="center", transform=axis.transAxes)
            continue

        sns.histplot(values, bins="auto", color="#4472C4", edgecolor="white", linewidth=0.5, ax=axis)
        if len(values) == 1:
            p50 = p95 = p99 = values[0]
        else:
            percentiles = quantiles(values, n=100, method="inclusive")
            p50, p95, p99 = percentiles[49], percentiles[94], percentiles[98]
        axis.axvline(p50, color="#C44E52", linestyle="--", linewidth=1, label=f"P50: {p50:.2f} ms")
        axis.axvline(p95, color="#8172B2", linestyle="-.", linewidth=1, label=f"P95: {p95:.2f} ms")
        axis.axvline(p99, color="#55A868", linestyle=":", linewidth=1, label=f"P99: {p99:.2f} ms")
        axis.yaxis.grid(False)
        axis.legend(frameon=False, fontsize=8)
        axis.text(0.02, 0.95, f"n = {len(values):,}", transform=axis.transAxes, ha="left", va="top")

    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export latency histograms from a Rerun recording to a single-page PDF."
    )
    parser.add_argument("recording", type=Path, help="Input Rerun .rrd file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF path (default: <recording>_latency_histogram.pdf)",
    )
    args = parser.parse_args()

    recording_path = args.recording.expanduser()
    if not recording_path.is_file():
        parser.error(f"recording file does not exist: {recording_path}")
    if recording_path.suffix.lower() != ".rrd":
        parser.error(f"expected a .rrd recording: {recording_path}")

    output_path = (
        args.output.expanduser()
        if args.output
        else recording_path.with_name(f"{recording_path.stem}_latency_histogram.pdf")
    )
    if output_path.resolve() == recording_path.resolve():
        parser.error("output PDF must not overwrite the input recording")
    if not output_path.parent.is_dir():
        parser.error(f"output directory does not exist: {output_path.parent}")

    samples = read_latency_samples(recording_path)
    save_histograms(samples, output_path, recording_path.name)

    for entity_path, title in SIGNALS:
        print(f"{title}: {len(samples[entity_path]):,} samples")
    print(f"Saved single-page PDF: {output_path}")


if __name__ == "__main__":
    main()
