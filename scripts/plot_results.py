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
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PRECISIONS = ("fp32", "fp16")
THREAD_COUNTS = (1, 2, 4)
RATES = (1.0, 2.0, 3.0)
CLASSES = ("car", "pedestrian")
MOT_SCORES = ("HOTA", "IDF1", "MOTP")
MOT_COUNTS = ("IDSW", "Frag")
LATENCY = "pipeline_end_to_end_latency_ms"
DROPS = "detector_frame_drop_count"


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
                    latency_rows.append((platform, precision, n_threads, percentile, samples["value"].quantile(quantile)))
            else:
                total = detector_drop_total(samples)
                drop_rows.append((platform, precision, n_threads, "Detector drops", total))

        mot = config[
            (config["source"] == "mot")
            & (config["sequence"] == "COMBINED")
            & (config["tracker"] == "bytetrack")
            & config["class"].isin(CLASSES)
            & config["metric"].isin((*MOT_SCORES, *MOT_COUNTS))
        ]
        for metric in (*MOT_SCORES, *MOT_COUNTS):
            values = mot[mot["metric"] == metric]
            if (
                len(values) != len(CLASSES)
                or set(values["class"]) != set(CLASSES)
                or not values["value"].map(math.isfinite).all()
            ):
                raise ValueError(f"{label}: expected one COMBINED {metric} per class")
            value = values["value"].mean()
            if metric in MOT_SCORES:
                value *= 100
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
        latency, drops, mot, output_dir, "executor", "configuration", panels, "n_threads",
        tuple(str(count) for count in THREAD_COUNTS),
    )


def load_effect_data(frame: pd.DataFrame, sequence: str | None = None) -> pd.DataFrame:
    """Sum per-recording detector drop maxima across matched FP32 configurations."""
    if sequence is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", sequence):
        raise ValueError("sequence must contain only letters, digits, underscores or hyphens")
    frame = frame[
        (frame["precision"] == "fp32") & frame["n_threads"].isin(THREAD_COUNTS) & frame["rate"].isin(RATES)
    ]
    if frame.empty:
        raise ValueError("no FP32 configurations found for the load comparison")
    platforms = sorted(frame["hardware"].dropna().unique())
    if not platforms or frame["hardware"].isna().any():
        raise ValueError("configuration hardware labels must not be missing")
    rows = []
    expected_sequences = None
    for platform, threads, rate in product(platforms, THREAD_COUNTS, RATES):
        config = frame[
            (frame["hardware"] == platform) & (frame["n_threads"] == threads) & (frame["rate"] == rate)
        ]
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
            x="rate", y="value", hue="n_threads", hue_order=tuple(str(count) for count in THREAD_COUNTS),
            marker="o", errorbar=None, ax=axis,
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
        figure.legend(handles, labels, title=legend, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(hue_values))
        figure.suptitle(title + " effect: " + ("latency" if aspect == "latency" else "detector frame drops"))
        figure.tight_layout(rect=(0, 0, 1, 0.83))
        save(figure, aspect)

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
    figure.suptitle(title + " effect: MOT KPIs")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    save(figure, "mot")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Evaluated results Parquet file")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--effect", choices=("precision", "platform", "executor", "load", "all"), default="precision")
    parser.add_argument("--sequence", help="Sequence ID for the load plot (default: all sequences)")
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


if __name__ == "__main__":
    main()
