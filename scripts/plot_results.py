#!/usr/bin/env python3
"""Plot comparisons from the evaluated results Parquet file."""

import argparse
import math
import re
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rerun as rr
import rerun_bindings
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

ROOT = Path(__file__).resolve().parents[1]
PRECISIONS = ("fp32", "fp16")
THREAD_COUNTS = (1, 2, 4)
RATES = (1.0, 2.0, 3.0)
CLASSES = ("car", "pedestrian")
MOT_SCORES = ("HOTA", "IDF1", "MOTP")
MOT_COUNTS = ("IDSW", "Frag")
LATENCY = "pipeline_end_to_end_latency_ms"
INFERENCE_LATENCY = "detector_inference_latency_ms"
DROPS = "detector_frame_drop_count"
SCALAR_COMPONENT = "Scalars:scalars"
BOX_WIDTH = 0.8


def sequence_inventory(config: pd.DataFrame, label: str) -> set[str]:
    sequences = config[(config["source"] == "sequence") & (config["metric"] == "input_frame_count")]
    if sequences.empty or sequences["sequence"].isna().any() or sequences["sequence"].duplicated().any():
        raise ValueError(f"{label}: missing or duplicate sequence inventory")
    return set(sequences["sequence"])


def pipeline_samples(config: pd.DataFrame, metric: str, sequence_ids: set[str], label: str) -> pd.DataFrame:
    samples = config[(config["source"] == "pipeline") & (config["metric"] == metric)]
    if (
        set(samples["sequence"]) != sequence_ids
        or samples["run"].isna().any()
        or not samples["value"].map(math.isfinite).all()
    ):
        raise ValueError(f"{label}: incomplete {metric} samples")
    if samples.groupby("sequence")["run"].nunique().ne(1).any():
        raise ValueError(f"{label}: expected one recording per sequence for {metric}")
    return samples


def detector_drop_total(samples: pd.DataFrame) -> float:
    return samples.groupby(["sequence", "run"])["value"].max().sum()


def combined_mot_values(config: pd.DataFrame, label: str) -> list[tuple[str, float]]:
    mot = config[
        (config["source"] == "mot")
        & (config["sequence"] == "COMBINED")
        & (config["tracker"] == "bytetrack")
        & config["class"].isin(CLASSES)
        & config["metric"].isin((*MOT_SCORES, *MOT_COUNTS))
    ]
    result = []
    for metric in (*MOT_SCORES, *MOT_COUNTS):
        values = mot[mot["metric"] == metric]
        if (
            len(values) != len(CLASSES)
            or set(values["class"]) != set(CLASSES)
            or not values["value"].map(math.isfinite).all()
        ):
            raise ValueError(f"{label}: expected one COMBINED {metric} per class")
        value = values["value"].mean()
        result.append((metric, value * 100 if metric in MOT_SCORES else value))
    return result


def precision_effect_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize the two-thread configurations for precision and platform comparisons."""
    latency, drops, mot = comparison_data(frame, (2,))
    return latency.drop(columns="n_threads"), drops.drop(columns="n_threads"), mot.drop(columns="n_threads")


def comparison_data(
    frame: pd.DataFrame, thread_counts: tuple[int, ...], precisions: tuple[str, ...] = PRECISIONS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize matched configurations without averaging per-sequence percentiles."""
    frame = frame[frame["n_threads"].isin(thread_counts) & (frame["rate"] == 1.0)]
    frame = frame[frame["precision"].isin(precisions)]
    if frame.empty:
        raise ValueError(f"no configurations found for precision={precisions}, rate=1.0 and n_threads={thread_counts}")

    platforms = sorted(frame["hardware"].dropna().unique())
    if not platforms or frame["hardware"].isna().any():
        raise ValueError("configuration hardware labels must not be missing")
    latency_rows, drop_rows, mot_rows = [], [], []
    expected_sequences = None
    for platform, precision, n_threads in product(platforms, precisions, thread_counts):
        config = frame[
            (frame["hardware"] == platform) & (frame["precision"] == precision) & (frame["n_threads"] == n_threads)
        ]
        label = f"{platform} {precision} n_threads={n_threads}"
        configs = config["config"].dropna().unique()
        if config.empty or config["config"].isna().any() or len(configs) != 1:
            raise ValueError(f"{label}: expected exactly one configuration")

        sequence_ids = sequence_inventory(config, label)
        if expected_sequences is None:
            expected_sequences = sequence_ids
        elif sequence_ids != expected_sequences:
            raise ValueError(f"{label}: sequence inventory differs between configurations")

        for metric in (LATENCY, DROPS):
            samples = pipeline_samples(config, metric, sequence_ids, label)
            if metric == LATENCY:
                for percentile, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
                    latency_rows.append(
                        (platform, precision, n_threads, percentile, samples["value"].quantile(quantile))
                    )
            else:
                total = detector_drop_total(samples)
                drop_rows.append((platform, precision, n_threads, "Detector drops", total))

        for metric, value in combined_mot_values(config, label):
            mot_rows.append((platform, precision, n_threads, metric, value))

    columns = ["hardware", "precision", "n_threads", "metric", "value"]
    return (
        pd.DataFrame(latency_rows, columns=columns),
        pd.DataFrame(drop_rows, columns=columns),
        pd.DataFrame(mot_rows, columns=columns),
    )


def plot_precision_effect(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one SVG and PNG per aspect, with platform panels and precision bars."""
    latency, drops, mot = precision_effect_data(frame)
    platforms = sorted(latency["hardware"].unique())
    return plot_effect(latency, drops, mot, output_dir, "precision", "hardware", platforms, "precision", PRECISIONS)


def plot_platform_effect(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one SVG and PNG per aspect, with precision panels and platform bars."""
    latency, drops, mot = precision_effect_data(frame)
    platforms = sorted(latency["hardware"].unique())
    if len(platforms) < 2:
        raise ValueError("platform effect requires at least two platforms")
    return plot_effect(latency, drops, mot, output_dir, "platform", "precision", PRECISIONS, "hardware", platforms)


def plot_executor_effect(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Compare thread counts within each platform at FP32 precision."""
    latency, drops, mot = comparison_data(frame, THREAD_COUNTS, ("fp32",))
    platforms = sorted(latency["hardware"].unique())
    panels = [f"{platform} / fp32" for platform in platforms]
    for data in (latency, drops, mot):
        data["configuration"] = data["hardware"] + " / " + data["precision"]
        data["n_threads"] = data["n_threads"].astype(str)
    return plot_effect(
        latency,
        drops,
        mot,
        output_dir,
        "executor",
        "configuration",
        panels,
        "n_threads",
        tuple(str(count) for count in THREAD_COUNTS),
    )


def load_effect_data(frame: pd.DataFrame, sequence: str | None = None) -> pd.DataFrame:
    """Sum per-recording detector drop maxima across matched FP32 configurations."""
    if sequence is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", sequence):
        raise ValueError("sequence must contain only letters, digits, underscores or hyphens")
    frame = frame[(frame["precision"] == "fp32") & frame["n_threads"].isin(THREAD_COUNTS) & frame["rate"].isin(RATES)]
    if frame.empty:
        raise ValueError("no FP32 configurations found for the load comparison")
    platforms = sorted(frame["hardware"].dropna().unique())
    if not platforms or frame["hardware"].isna().any():
        raise ValueError("configuration hardware labels must not be missing")
    rows = []
    expected_sequences = None
    for platform, threads, rate in product(platforms, THREAD_COUNTS, RATES):
        config = frame[(frame["hardware"] == platform) & (frame["n_threads"] == threads) & (frame["rate"] == rate)]
        label = f"{platform} fp32 n_threads={threads} rate={rate}"
        if config.empty or config["config"].isna().any() or config["config"].nunique() != 1:
            raise ValueError(f"{label}: expected exactly one configuration")
        sequence_ids = sequence_inventory(config, label)
        if sequence is not None:
            if sequence not in sequence_ids:
                raise ValueError(f"{label}: sequence {sequence!r} is missing")
            sequence_ids = {sequence}
            config = config[config["sequence"] == sequence]
        if expected_sequences is None:
            expected_sequences = sequence_ids
        elif sequence_ids != expected_sequences:
            raise ValueError(f"{label}: sequence inventory differs between configurations")
        samples = pipeline_samples(config, DROPS, sequence_ids, label)
        rows.append((platform, threads, rate, detector_drop_total(samples)))
    return pd.DataFrame(rows, columns=["hardware", "n_threads", "rate", "value"])


def plot_load_effect(frame: pd.DataFrame, output_dir: Path, sequence: str | None = None) -> list[Path]:
    """Compare detector frame drops across replay rates and thread counts."""
    data = load_effect_data(frame, sequence)
    data["n_threads"] = data["n_threads"].astype(str)
    platforms = sorted(data["hardware"].unique())
    sns.set_theme(style="darkgrid", palette="deep")
    figure, axes = plt.subplots(1, len(platforms), figsize=(5 * len(platforms), 4), squeeze=False, sharey=True)
    for axis, platform in zip(axes[0], platforms):
        sns.lineplot(
            data=data[data["hardware"] == platform],
            x="rate",
            y="value",
            hue="n_threads",
            hue_order=tuple(str(count) for count in THREAD_COUNTS),
            marker="o",
            errorbar=None,
            ax=axis,
        )
        axis.set(title=platform, xlabel="Replay rate", ylabel="Total detector dropped frames", xticks=RATES)
        axis.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, title="Threads", loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3)
    sequence_label = f"sequence {sequence}" if sequence is not None else "all sequences"
    figure.suptitle(f"Load effect: detector frame drops (FP32, {sequence_label})")
    figure.tight_layout(rect=(0, 0, 1, 0.83))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{sequence}" if sequence is not None else ""
    paths = [output_dir / f"load_effect_drops{suffix}.{extension}" for extension in ("svg", "png")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    return paths


def load_platform_configurations(frame: pd.DataFrame) -> list[tuple[str, float, pd.DataFrame, set[str]]]:
    """Select matched FP32 configurations at two threads across platforms and rates."""
    frame = frame[(frame["precision"] == "fp32") & (frame["n_threads"] == 2) & frame["rate"].isin(RATES)]
    platforms = sorted(frame["hardware"].dropna().unique())
    if frame["hardware"].isna().any() or len(platforms) != 2:
        raise ValueError("load-platform interaction requires exactly two platforms with valid hardware labels")
    configurations = []
    expected_sequences = None
    for platform, rate in product(platforms, RATES):
        config = frame[(frame["hardware"] == platform) & (frame["rate"] == rate)]
        label = f"{platform} fp32 n_threads=2 rate={rate}"
        if config.empty or config["config"].isna().any() or config["config"].nunique() != 1:
            raise ValueError(f"{label}: expected exactly one configuration")
        sequence_ids = sequence_inventory(config, label)
        if expected_sequences is None:
            expected_sequences = sequence_ids
        elif sequence_ids != expected_sequences:
            raise ValueError(f"{label}: sequence inventory differs between configurations")
        configurations.append((platform, rate, config, sequence_ids))
    return configurations


def load_platform_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Select matched FP32 inference samples at two threads across platforms and rates."""
    samples_by_config = []
    for platform, rate, config, sequence_ids in load_platform_configurations(frame):
        label = f"{platform} fp32 n_threads=2 rate={rate}"
        samples = pipeline_samples(config, INFERENCE_LATENCY, sequence_ids, label)
        if (samples["value"] < 0).any():
            raise ValueError(f"{label}: inference latencies must be non-negative")
        samples_by_config.append(samples)
    return pd.concat(samples_by_config, ignore_index=True)


def load_platform_mot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize class-averaged MOT metrics for matched platform and rate configurations."""
    rows = []
    for platform, rate, config, _ in load_platform_configurations(frame):
        label = f"{platform} fp32 n_threads=2 rate={rate}"
        for metric, value in combined_mot_values(config, label):
            rows.append((platform, f"{rate:g}x", metric, value))
    return pd.DataFrame(rows, columns=["hardware", "rate", "metric", "value"])


def plot_load_platform_mot(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Compare MOT metrics by platform and replay rate at FP32 and two threads."""
    mot = load_platform_mot_data(frame)
    figure = plot_mot_effect(
        mot, "hardware", sorted(mot["hardware"].unique()), "rate",
        tuple(f"{rate:g}x" for rate in RATES), "Replay rate",
        "Load x platform: MOT KPIs (FP32, 2 threads, all sequences)",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"load_platform_mot.{extension}" for extension in ("svg", "png")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    return paths


def plot_load_platform_effect(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot pooled inference latency distributions for both platforms at each replay rate."""
    data = load_platform_data(frame)
    platforms = sorted(data["hardware"].unique())
    maximum = max(1.0, data["value"].max())
    bins = [0.0, 1.0] if maximum == 1.0 else [0.0] + [math.exp(math.log(maximum) * i / 300) for i in range(301)]
    bins[-1] = maximum
    sns.set_theme(style="darkgrid", palette="deep")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, squeeze=False)
    for row, platform in enumerate(platforms):
        for column, rate in enumerate(RATES):
            axis = axes[row, column]
            samples = data[(data["hardware"] == platform) & (data["rate"] == rate)]
            sns.histplot(data=samples, x="value", bins=bins, stat="percent", ax=axis)
            axis.set(
                title=f"{platform} / {rate:g}x",
                xlabel="Inference latency [ms] (symlog)",
                ylabel="Samples per bin [%] (log)",
            )
    axes[0, 0].set_xscale("symlog", linthresh=1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlim(0, maximum * 1.05)
    figure.suptitle("Load x platform: inference latency (FP32, 2 threads, all sequences)")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"load_platform_inference_latency.{extension}" for extension in ("svg", "png")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    return paths


def draw_load_platform_boxplot(data: pd.DataFrame, axis) -> None:
    sns.boxplot(
        data=data,
        x="rate",
        y="value",
        hue="hardware",
        order=RATES,
        hue_order=sorted(data["hardware"].unique()),
        width=BOX_WIDTH,
        whis=1.5,
        flierprops={
            "marker": "o",
            "markersize": 2,
            "markerfacecolor": "none",
            "markeredgecolor": "0.25",
            "markeredgewidth": 0.5,
            "alpha": 0.3,
        },
        ax=axis,
    )
    axis.set_yscale("symlog", linthresh=5.0)
    axis.set(xlabel="Replay rate", ylabel="Inference latency [ms] (symlog)")


def exclude_first_inferences(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exactly one earliest inference sample from each recording."""
    if (
        "timestamp_ms" not in data
        or data["timestamp_ms"].isna().any()
        or not data["timestamp_ms"].map(math.isfinite).all()
    ):
        raise ValueError("inference timestamps must be present and finite")
    keys = ["config", "sequence", "run"]
    first = data["timestamp_ms"].eq(data.groupby(keys)["timestamp_ms"].transform("min"))
    removed = data[first]
    if removed.groupby(keys).size().ne(1).any():
        raise ValueError("cannot identify the first inference: multiple samples share the earliest timestamp")
    retained = data[~first]
    if len(retained.groupby(["rate", "hardware"])) != len(data.groupby(["rate", "hardware"])):
        raise ValueError("each platform and rate must retain at least one inference sample")
    return retained, removed


def plot_load_platform_boxplot(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Compare pooled inference latency after excluding each recording's first inference."""
    data, removed = exclude_first_inferences(load_platform_data(frame))
    summary = removed.groupby(["rate", "hardware"])["value"].agg(count="size", maximum="max")
    sns.set_theme(style="darkgrid", palette="deep")
    figure, axis = plt.subplots(figsize=(13, 6))
    draw_load_platform_boxplot(data, axis)
    scale = axis.yaxis.get_transform()
    data_maximum = max(1.0, data["value"].max())
    maximum = float(scale.inverted().transform(scale.transform(data_maximum) / 0.8))
    if not math.isfinite(maximum):
        plt.close(figure)
        raise ValueError("inference latency is too large to plot")
    axis.set_ylim(0, maximum)
    ticks = [0.0, 1.0, 5.0]
    tick_base = 10.0
    while tick_base <= maximum:
        ticks.extend((tick_base, 2 * tick_base, 5 * tick_base))
        tick_base *= 10
    axis.set_yticks([tick for tick in ticks if tick <= maximum])
    axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    handles, labels = axis.get_legend_handles_labels()
    legend = axis.get_legend()
    if legend is not None:
        legend.remove()
    figure.legend(handles, labels, title="Platform", loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=2)
    platforms = sorted(data["hardware"].unique())
    equal_counts = summary["count"].nunique() == 1
    for position, rate in zip(axis.get_xticks(), RATES, strict=True):
        for index, hardware in enumerate(platforms):
            row = summary.loc[(rate, hardware)]
            label = f"{row['maximum']:.2f}"
            if not equal_counts:
                label += f"\nn={int(row['count'])}"
            offset = BOX_WIDTH * ((index + 0.5) / len(platforms) - 0.5)
            axis.text(
                position + offset, 0.9, label, transform=axis.get_xaxis_transform(),
                ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7"},
            )
    count_text = f"n={int(summary['count'].iloc[0])} per box; " if equal_counts else ""
    figure.suptitle(
        "Load x platform: inference latency (FP32, 2 threads, all sequences)\n"
        f"First inference excluded: {count_text}labels show excluded maximum [ms]"
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.79)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"load_platform_inference_boxplot.{extension}" for extension in ("svg", "png")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    return paths


def load_platform_extreme_limit(data: pd.DataFrame) -> float:
    group_p995 = data.groupby(["rate", "hardware"])["value"].quantile(0.995)
    return max(1.0, math.ceil(group_p995.max()))


def load_platform_extremes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Find samples above the extreme limit and their position within each recording."""
    data = load_platform_data(frame)
    limit = load_platform_extreme_limit(data)
    recordings = frame[(frame["source"] == "pipeline") & frame["config"].isin(data["config"].unique())]
    if (
        recordings["timestamp_ms"].isna().any()
        or not recordings["timestamp_ms"].map(math.isfinite).all()
        or recordings["sequence"].isna().any()
        or recordings["run"].isna().any()
    ):
        raise ValueError("pipeline recording timestamps, sequences and run labels must be present and finite")
    keys = ["config", "sequence", "run"]
    if recordings.groupby(["config", "sequence"])["run"].nunique().ne(1).any():
        raise ValueError("expected one pipeline recording per configuration and sequence")
    starts = recordings.groupby(keys)["timestamp_ms"].min().rename("first_pipeline_ms").reset_index()
    first_inference = data.groupby(keys)["timestamp_ms"].transform("min")
    extremes = data[data["value"] > limit].copy()
    extremes["first_inference"] = extremes["timestamp_ms"].eq(first_inference.loc[extremes.index])
    extremes = extremes.merge(starts, on=keys, how="left", validate="many_to_one")
    if extremes["first_pipeline_ms"].isna().any():
        raise ValueError("missing first pipeline timestamp for an extreme inference sample")
    extremes["elapsed_seconds"] = (extremes["timestamp_ms"] - extremes["first_pipeline_ms"]) / 1000
    return data, extremes, limit


def first_image_timestamp(sequence_path: Path) -> int:
    """Read the first source image's header timestamp, in nanoseconds."""
    if not sequence_path.is_file():
        raise FileNotFoundError(f"source sequence does not exist: {sequence_path}")
    with Reader(sequence_path) as reader:
        connections = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/CompressedImage"]
        if len(connections) != 1:
            raise ValueError(f"{sequence_path}: expected exactly one compressed image connection")
        message = next(reader.messages(connections=connections), None)
        if message is None:
            raise ValueError(f"{sequence_path}: no source images")
        connection, _, raw = message
        image = get_typestore(Stores.ROS2_HUMBLE).deserialize_cdr(raw, connection.msgtype)
        return image.header.stamp.sec * 1_000_000_000 + image.header.stamp.nanosec


def recording_inference_timestamps(recording_path: Path, export_end_ms: int) -> list[tuple[int, float, int]]:
    """Read inference values and both timelines, reconstructing the exported millisecond timestamps."""
    if not recording_path.is_file():
        raise FileNotFoundError(f"recording does not exist: {recording_path}")
    reader = rerun_bindings.RrdReaderInternal(str(recording_path))
    stream = rr.chunk.LazyChunkStream(reader.stream()).filter(components=SCALAR_COMPONENT)
    latest = None
    inference = []
    for chunk in stream:
        batch = chunk.to_record_batch()
        if "steady_elapsed" not in batch.schema.names:
            raise ValueError(f"{recording_path}: scalar chunk lacks the steady_elapsed timeline")
        is_inference = chunk.entity_path.strip("/") == "detector/inference_latency_ms"
        if is_inference and "ros2_timestamp" not in batch.schema.names:
            raise ValueError(f"{recording_path}: inference chunk lacks the ros2_timestamp timeline")
        steady = batch.column("steady_elapsed")
        values = batch.column(SCALAR_COMPONENT)
        source = batch.column("ros2_timestamp") if is_inference else None
        for index in range(batch.num_rows):
            timestamp = steady[index]
            row_values = values[index].as_py()
            if not timestamp.is_valid or row_values is None:
                continue
            finite_values = [float(value) for value in row_values if value is not None and math.isfinite(float(value))]
            if not finite_values:
                continue
            timestamp_ns = timestamp.value
            latest = timestamp_ns if latest is None else max(latest, timestamp_ns)
            if is_inference:
                if source is None or not source[index].is_valid:
                    raise ValueError(f"{recording_path}: inference sample lacks a source-frame timestamp")
                inference.extend((timestamp_ns, value, source[index].value) for value in finite_values)
    if latest is None or not inference:
        raise ValueError(f"{recording_path}: no finite inference samples")
    # The exporter shifts elapsed timestamps so the last scalar lands at its export time.
    return [
        (export_end_ms - (latest - timestamp) // 1_000_000, value, source) for timestamp, value, source in inference
    ]


def load_platform_sequence_extremes(
    frame: pd.DataFrame, runs_dir: Path, sequences_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Match exported samples to their source-image timestamps and sequence starts."""
    data, _, limit = load_platform_extremes(frame)
    keys = ["config", "sequence", "run"]
    recordings = frame[(frame["source"] == "pipeline") & frame["config"].isin(data["config"].unique())]
    export_ends = recordings.groupby(keys)["timestamp_ms"].max()
    starts = {}
    matched = []
    for (config, sequence, run), group in data.groupby(keys):
        if any(
            not isinstance(part, str) or Path(part).name != part or part in ("", ".", "..")
            for part in (config, sequence, run)
        ):
            raise ValueError("configuration, sequence and run labels must be single path components")
        if sequence not in starts:
            starts[sequence] = first_image_timestamp(sequences_dir / sequence / f"{sequence}.mcap")
        recording_path = runs_dir / config / run / "recording.rrd"
        raw = sorted(recording_inference_timestamps(recording_path, int(export_ends.loc[(config, sequence, run)])))
        exported = group.sort_values(["timestamp_ms", "value"], kind="stable").copy()
        if len(raw) != len(exported):
            raise ValueError(f"{recording_path}: inference sample count differs from Parquet")
        source_seconds = []
        for (timestamp, value, source_ns), (exported_timestamp, exported_value) in zip(
            raw, exported[["timestamp_ms", "value"]].itertuples(index=False, name=None), strict=True
        ):
            if timestamp != exported_timestamp or not math.isclose(value, exported_value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{recording_path}: inference samples differ from Parquet")
            if source_ns < starts[sequence]:
                raise ValueError(f"{recording_path}: inference frame precedes the source sequence")
            source_seconds.append((source_ns - starts[sequence]) / 1_000_000_000)
        exported["elapsed_seconds"] = source_seconds
        exported["first_inference"] = exported["timestamp_ms"].eq(group["timestamp_ms"].min())
        matched.append(exported)
    samples = pd.concat(matched, ignore_index=True)
    return data, samples[samples["value"] > limit].copy(), limit


def plot_extreme_times(data: pd.DataFrame, extremes: pd.DataFrame, limit: float, description: str) -> plt.Figure:
    """Plot"""
    platforms = sorted(data["hardware"].unique())
    maximum = max(1.0, extremes["elapsed_seconds"].max()) if not extremes.empty else 1.0
    bins = [i / 10 for i in range(11)] + [1.25, 1.5, 2, 3, 5, 10, 20, 40, 80]
    while bins[-1] < maximum:
        bins.append(bins[-1] * 2)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, squeeze=False)
    for row, platform in enumerate(platforms):
        for column, rate in enumerate(RATES):
            samples = extremes[(extremes["hardware"] == platform) & (extremes["rate"] == rate)]
            label = f"{platform} / {rate:g}x (n={len(samples)}, first inference={samples['first_inference'].sum()})"
            axis = axes[row, column]
            if not samples.empty:
                sns.histplot(data=samples, x="elapsed_seconds", bins=bins, ax=axis)
            axis.set(title=label, xlabel=f"Seconds since {description} (symlog)", ylabel="Count per bin")
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].set_xscale("symlog", linthresh=1)
    time_limit = max(1.0, maximum * 1.05)
    axes[0, 0].set_xlim(0, time_limit)
    axes[0, 0].set_xticks([tick for tick in (0, 0.5, 1, 2, 5, 10, 20, 50, 100) if tick <= time_limit])
    axes[0, 0].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    figure.suptitle(f"Inference samples > {limit:g} ms: time since {description} (FP32, 2 threads)")
    figure.tight_layout()
    return figure


def plot_load_platform_sequence_time(
    frame: pd.DataFrame, output_dir: Path, runs_dir: Path, sequences_dir: Path
) -> list[Path]:
    data, extremes, limit = load_platform_sequence_extremes(frame, runs_dir, sequences_dir)
    sns.set_theme(style="darkgrid", palette="deep")
    figure = plot_extreme_times(data, extremes, limit, "source sequence start")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"load_platform_extreme_sequence_times.{extension}" for extension in ("svg", "png")]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    return paths


def plot_load_platform_extremes(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot when and in which sequences extreme inference samples were logged."""
    data, extremes, limit = load_platform_extremes(frame)
    platforms = sorted(data["hardware"].unique())
    sequences = sorted(data["sequence"].unique())
    sns.set_theme(style="darkgrid", palette="deep")
    time_figure = plot_extreme_times(data, extremes, limit, "first pipeline observation")
    sequence_figure, sequence_axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, squeeze=False)
    for row, platform in enumerate(platforms):
        for column, rate in enumerate(RATES):
            samples = extremes[(extremes["hardware"] == platform) & (extremes["rate"] == rate)]
            label = f"{platform} / {rate:g}x (n={len(samples)}, first inference={samples['first_inference'].sum()})"
            counts = samples["sequence"].value_counts().reindex(sequences, fill_value=0)
            sequence_axis = sequence_axes[row, column]
            sns.barplot(x=counts.index, y=counts.to_numpy(), color=sns.color_palette("deep")[0], ax=sequence_axis)
            sequence_axis.set(title=label, xlabel="Sequence", ylabel="Extreme inference samples")
            sequence_axis.tick_params(axis="x", labelrotation=90, labelsize=8)
            sequence_axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    sequence_figure.suptitle(f"Inference samples > {limit:g} ms: counts by sequence (FP32, 2 threads)")
    sequence_figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for figure, name in ((time_figure, "times"), (sequence_figure, "sequences")):
        for extension in ("svg", "png"):
            path = output_dir / f"load_platform_extreme_{name}.{extension}"
            figure.savefig(path, bbox_inches="tight", dpi=200)
            paths.append(path)
        plt.close(figure)
    return paths


def plot_mot_effect(
    mot: pd.DataFrame,
    panel: str,
    panel_values: Sequence[str],
    hue: str,
    hue_values: Sequence[str],
    legend: str,
    title: str,
) -> plt.Figure:
    sns.set_theme(style="darkgrid", palette="deep")
    figure, axes = plt.subplots(2, len(panel_values), figsize=(5 * len(panel_values), 7), squeeze=False, sharey="row")
    for column, panel_value in enumerate(panel_values):
        for row, (metrics, ylabel) in enumerate(((MOT_SCORES, "Score [%]"), (MOT_COUNTS, "Mean count per class"))):
            axis = axes[row, column]
            sns.barplot(
                data=mot[(mot[panel] == panel_value) & mot["metric"].isin(metrics)],
                x="metric",
                y="value",
                hue=hue,
                order=metrics,
                hue_order=hue_values,
                errorbar=None,
                ax=axis,
            )
            axis.set(title=panel_value if row == 0 else "", xlabel="", ylabel=ylabel)
            axis.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, title=legend, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(hue_values))
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return figure


def plot_effect(
    latency: pd.DataFrame,
    drops: pd.DataFrame,
    mot: pd.DataFrame,
    output_dir: Path,
    effect: str,
    panel: str,
    panel_values: Sequence[str],
    hue: str,
    hue_values: Sequence[str],
) -> list[Path]:
    sns.set_theme(style="darkgrid", palette="deep")
    title = "Executor threading" if effect == "executor" else effect.capitalize()
    legend = "Threads" if effect == "executor" else effect.capitalize()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(figure, aspect: str) -> None:
        for extension in ("svg", "png"):
            path = output_dir / f"{effect}_effect_{aspect}.{extension}"
            figure.savefig(path, bbox_inches="tight", dpi=200)
            outputs.append(path)
        plt.close(figure)

    for aspect, data, ylabel, order in (
        ("latency", latency, "End-to-end latency [ms]", ("p50", "p95", "p99")),
        ("drops", drops, "Total dropped frames", ("Detector drops",)),
    ):
        figure, axes = plt.subplots(
            1, len(panel_values), figsize=(5 * len(panel_values), 4), squeeze=False, sharey=True
        )
        for axis, panel_value in zip(axes[0], panel_values):
            sns.barplot(
                data=data[data[panel] == panel_value],
                x="metric",
                y="value",
                hue=hue,
                order=order,
                hue_order=hue_values,
                errorbar=None,
                ax=axis,
            )
            axis.set(title=panel_value, xlabel="", ylabel=ylabel)
            axis.get_legend().remove()
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles, labels, title=legend, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(hue_values)
        )
        figure.suptitle(title + " effect: " + ("latency" if aspect == "latency" else "detector frame drops"))
        figure.tight_layout(rect=(0, 0, 1, 0.83))
        save(figure, aspect)

    figure = plot_mot_effect(mot, panel, panel_values, hue, hue_values, legend, title + " effect: MOT KPIs")
    save(figure, "mot")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Evaluated results Parquet file")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument(
        "--effect",
        choices=(
            "precision",
            "platform",
            "executor",
            "load",
            "load-platform",
            "load-platform-mot",
            "load-platform-box",
            "load-platform-extremes",
            "load-platform-sequence-time",
            "all",
        ),
        default="precision",
    )
    parser.add_argument("--sequence", help="Sequence ID for the load plot (default: all sequences)")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help="Root directory of Rerun recordings")
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=ROOT / "data/kitti/left_color/training/image_02_mcap",
        help="Root directory of source-sequence MCAP files",
    )
    args = parser.parse_args()
    if args.sequence is not None and args.effect not in ("load", "all"):
        parser.error("--sequence applies only to --effect load or all")
    frame = pd.read_parquet(args.parquet)
    if args.effect in ("precision", "all"):
        for path in plot_precision_effect(frame, args.output_dir):
            print(f"Wrote {path}")
    if args.effect in ("platform", "all"):
        for path in plot_platform_effect(frame, args.output_dir):
            print(f"Wrote {path}")
    if args.effect in ("executor", "all"):
        for path in plot_executor_effect(frame, args.output_dir):
            print(f"Wrote {path}")
    if args.effect in ("load", "all"):
        for path in plot_load_effect(frame, args.output_dir, args.sequence):
            print(f"Wrote {path}")
    if args.effect in ("load-platform", "all"):
        for path in plot_load_platform_effect(frame, args.output_dir):
            print(f"Wrote {path}")
    if args.effect in ("load-platform-mot", "all"):
        for path in plot_load_platform_mot(frame, args.output_dir):
            print(f"Wrote {path}")
    if args.effect in ("load-platform-box", "all"):
        for path in plot_load_platform_boxplot(frame, args.output_dir):
            print(f"Wrote {path}")
    if args.effect in ("load-platform-extremes", "all"):
        for path in plot_load_platform_extremes(frame, args.output_dir):
            print(f"Wrote {path}")
    if args.effect == "load-platform-sequence-time":
        for path in plot_load_platform_sequence_time(frame, args.output_dir, args.runs_dir, args.sequences_dir):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
