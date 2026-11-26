#!/usr/bin/env python3
"""Combine evaluated run CSVs and sequence lengths into a single Parquet file."""

import argparse
import csv
import fnmatch
import math
import os
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 50_000
PIPELINE_COLUMNS = [
    "timestamp_ms",
    "run",
    "metric",
    "hardware",
    "precision",
    "n_threads",
    "rate",
    "sequence",
    "config",
    "value",
]
PIPELINE_TIMELINE_COLUMNS = ["ros2_timestamp_ns", "steady_elapsed_ns"]
MOT_COLUMNS = [
    "timestamp_ms",
    "hardware",
    "precision",
    "n_threads",
    "rate",
    "config",
    "tracker",
    "sequence",
    "class",
    "metric",
    "value",
]
DEVICE_METRICS = [
    "cpu_temp_c",
    "cpu_usage_pct",
    "gpu_temp_c",
    "gpu_usage_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "aux_temp_c",
    "emc_usage_pct",
    "ram_used_mb",
    "ram_total_mb",
    "swap_used_mb",
    "power_mw",
    "fan_rpm",
]
DEVICE_COLUMNS = ["timestamp_ms", *DEVICE_METRICS, "config", "hardware"]
SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("source_file", pa.string()),
        ("timestamp_ms", pa.int64()),
        ("run", pa.string()),
        ("hardware", pa.string()),
        ("precision", pa.string()),
        ("n_threads", pa.int64()),
        ("rate", pa.float64()),
        ("sequence", pa.string()),
        ("tracker", pa.string()),
        ("class", pa.string()),
        ("config", pa.string()),
        ("metric", pa.string()),
        ("value", pa.float64()),
        ("ros2_timestamp_ns", pa.int64()),
        ("steady_elapsed_ns", pa.int64()),
    ]
)


def load_metadata(config_dir: Path) -> dict:
    path = config_dir / "metadata.yaml"
    with path.open(encoding="utf-8") as file:
        metadata = yaml.safe_load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    for key in ("config", "hardware", "precision"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"{path}: {key} must be a non-empty string")
    if metadata["config"] != config_dir.name:
        raise ValueError(f"{path}: config must match directory name {config_dir.name!r}")
    threads = metadata.get("n_threads")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError(f"{path}: n_threads must be a positive integer")
    rate = metadata.get("rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{path}: rate must be a positive finite number")
    sequences = metadata.get("sequences")
    if (
        not isinstance(sequences, list)
        or not sequences
        or any(not isinstance(s, str) or not s for s in sequences)
        or len(set(sequences)) != len(sequences)
        or "COMBINED" in sequences
    ):
        raise ValueError(f"{path}: sequences must be a non-empty list of unique string IDs")
    return metadata


def sequence_lengths(config_dir: Path, metadata: dict) -> dict[str, int]:
    path = config_dir / "eval" / "evaluate_tracking.seqmap.training"
    counts = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) != 4 or parts[0] in counts:
                raise ValueError(f"{path}: invalid or duplicate seqmap entry: {line.strip()!r}")
            try:
                count = int(parts[3])
            except ValueError as error:
                raise ValueError(f"{path}: invalid frame count in {line.strip()!r}") from error
            if count <= 0:
                raise ValueError(f"{path}: frame count must be positive for {parts[0]}")
            counts[parts[0]] = count
    if set(counts) != set(metadata["sequences"]):
        raise ValueError(f"{path}: seqmap sequences do not match metadata")
    return counts


def check_labels(frame: pd.DataFrame, metadata: dict, path: Path, source: str) -> None:
    expected = {key: metadata[key] for key in ("config", "hardware", "precision", "n_threads", "rate")}
    for key, value in expected.items():
        if key not in frame:
            continue
        actual = frame[key]
        if key == "hardware" and source == "hardware" and value == "rtx3090":
            valid = actual.isin(["pc", "rtx3090"])
        else:
            valid = actual.eq(value)
        if not valid.all():
            raise ValueError(f"{path}: {key} does not match metadata {value!r}")


def write_frame(
    writer: pq.ParquetWriter, frame: pd.DataFrame, source: str, path: Path, runs_dir: Path, metadata: dict
) -> int:
    frame = frame.copy()
    frame["source"] = source
    frame["source_file"] = path.relative_to(runs_dir).as_posix()
    for key in ("config", "hardware", "precision", "n_threads", "rate"):
        frame[key] = metadata[key]
    for field in SCHEMA:
        if field.name not in frame:
            frame[field.name] = None
    table = pa.Table.from_pandas(frame[SCHEMA.names], schema=SCHEMA, preserve_index=False, safe=True)
    writer.write_table(table)
    return len(frame)


def read_batches(path: Path, columns: list[str], *, header: bool):
    kwargs = {
        "dtype": {name: "string" for name in columns if name not in DEVICE_METRICS},
        "chunksize": BATCH_SIZE,
        "keep_default_na": False,
    }
    if header:
        with path.open(newline="", encoding="utf-8") as file:
            if next(csv.reader(file), None) != columns:
                raise ValueError(f"{path}: unexpected CSV header")
    else:
        kwargs.update({"header": None, "names": columns})
    with pd.read_csv(path, **kwargs) as reader:
        for frame in reader:
            if frame.empty:
                continue
            for column in ("timestamp_ms", "n_threads", "rate", "value"):
                if column in frame:
                    frame[column] = pd.to_numeric(frame[column], errors="raise")
            for column in PIPELINE_TIMELINE_COLUMNS:
                if column in frame:
                    frame[column] = pd.array(frame[column].replace("", pd.NA), dtype="Int64")
            yield frame


def write_config(writer: pq.ParquetWriter, config_dir: Path, runs_dir: Path) -> dict[str, int]:
    metadata = load_metadata(config_dir)
    lengths = sequence_lengths(config_dir, metadata)
    pipeline_dir = config_dir / "eval" / "pipeline"

    files_by_sequence: dict[str, list[Path]] = {sequence: [] for sequence in lengths}
    for path in sorted(pipeline_dir.glob("kpi_*.csv")):
        run_id = path.stem.removeprefix("kpi_")
        if run_id in lengths:
            sequence = run_id
        else:
            timestamp, separator, sequence = run_id.rpartition("_")
            if not separator or not timestamp or sequence not in lengths:
                raise ValueError(f"{path}: pipeline CSV filename does not match metadata sequences")
        files_by_sequence[sequence].append(path)
    if any(not paths for paths in files_by_sequence.values()):
        raise ValueError(f"{pipeline_dir}: pipeline CSV files do not match metadata sequences")

    pipeline_files = []
    for sequence, paths in files_by_sequence.items():
        legacy_path = pipeline_dir / f"kpi_{sequence}.csv"
        if len(paths) > 1 and legacy_path in paths:
            with legacy_path.open(newline="", encoding="utf-8") as file:
                first_row = next(csv.reader(file), None)
            if first_row and len(first_row) > 1 and pipeline_dir / f"kpi_{first_row[1]}.csv" in paths:
                print(f"Warning: ignoring duplicate legacy CSV {legacy_path}", file=sys.stderr)
                paths.remove(legacy_path)
        pipeline_files.extend((path, sequence) for path in paths)

    totals = {"pipeline": 0, "mot": 0, "hardware": 0, "sequence": 0}
    for path, sequence in [*pipeline_files, (config_dir / "eval" / "tracker_kpi.csv", None)]:
        source = "pipeline" if sequence is not None else "mot"
        if source == "pipeline":
            with path.open(newline="", encoding="utf-8") as file:
                first_row = next(csv.reader(file), [])
            if len(first_row) == len(PIPELINE_COLUMNS):
                columns = PIPELINE_COLUMNS
            elif len(first_row) == len(PIPELINE_COLUMNS) + len(PIPELINE_TIMELINE_COLUMNS):
                columns = [*PIPELINE_COLUMNS, *PIPELINE_TIMELINE_COLUMNS]
            else:
                raise ValueError(f"{path}: unexpected pipeline CSV column count: {len(first_row)}")
        else:
            columns = MOT_COLUMNS
        before = totals[source]
        runs = set()
        with closing(read_batches(path, columns, header=False)) as batches:
            for frame in batches:
                check_labels(frame, metadata, path, source)
                valid_sequences = {sequence} if sequence is not None else set(lengths) | {"COMBINED"}
                if not frame["sequence"].isin(valid_sequences).all():
                    raise ValueError(f"{path}: invalid sequence")
                for name in ["run", "metric"] if source == "pipeline" else ["tracker", "class", "metric"]:
                    if frame[name].eq("").any():
                        raise ValueError(f"{path}: {name} must not be empty")
                if not frame["value"].map(math.isfinite).all():
                    raise ValueError(f"{path}: metric values must be finite")
                if source == "pipeline":
                    runs.update(frame["run"].unique())
                totals[source] += write_frame(writer, frame, source, path, runs_dir, metadata)
        if totals[source] == before:
            raise ValueError(f"{path}: CSV is empty")
        if source == "pipeline" and len(runs) != 1:
            raise ValueError(f"{path}: expected exactly one run ID")
        if sequence is not None and path.stem != f"kpi_{sequence}" and runs != {path.stem.removeprefix("kpi_")}:
            raise ValueError(f"{path}: run ID does not match pipeline CSV filename")

    path = config_dir / "eval" / "device_metrics.csv"
    if path.exists():
        with closing(read_batches(path, DEVICE_COLUMNS, header=True)) as batches:
            for frame in batches:
                check_labels(frame, metadata, path, "hardware")
                for column in DEVICE_METRICS:
                    frame[column] = pd.to_numeric(frame[column].replace({"nan": None, "": None}), errors="raise")
                    if not frame[column].dropna().map(math.isfinite).all():
                        raise ValueError(f"{path}: {column} contains a non-finite value")
                frame = frame.melt(
                    id_vars=["timestamp_ms", "hardware", "config"],
                    value_vars=DEVICE_METRICS,
                    var_name="metric",
                    value_name="value",
                )
                totals["hardware"] += write_frame(writer, frame, "hardware", path, runs_dir, metadata)
        if not totals["hardware"]:
            raise ValueError(f"{path}: CSV is empty")
    else:
        print(f"Warning: {path} is missing; hardware measurements are unavailable", file=sys.stderr)

    path = config_dir / "eval" / "evaluate_tracking.seqmap.training"
    frame = pd.DataFrame({"sequence": list(lengths), "value": list(lengths.values())})
    frame["metric"] = "input_frame_count"
    totals["sequence"] = write_frame(writer, frame, "sequence", path, runs_dir, metadata)
    return totals


def build_parquet(runs_dir: Path, pattern: str, output: Path) -> dict[str, int]:
    runs_dir = runs_dir.resolve()
    configs = sorted(path for path in runs_dir.iterdir() if path.is_dir() and fnmatch.fnmatchcase(path.name, pattern))
    if not configs:
        raise ValueError(f"no configuration directories matching {pattern!r} in {runs_dir}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = {"pipeline": 0, "mot": 0, "hardware": 0, "sequence": 0}
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:
            for config_dir in configs:
                print(f"Processing {config_dir.name}")
                for source, count in write_config(writer, config_dir, runs_dir).items():
                    totals[source] += count
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help="Directory containing run configurations")
    parser.add_argument("--config", default="*", help="Configuration directory name or quoted glob (default: all)")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "results" / "results.parquet")
    args = parser.parse_args()
    if not args.config or "/" in args.config or "\\" in args.config or args.config in {".", ".."}:
        parser.error("--config must be a name or glob without path separators")
    try:
        totals = build_parquet(args.runs_dir, args.config, args.output)
    except (OSError, ValueError, pa.ArrowException, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}: " + ", ".join(f"{source}={count:,}" for source, count in totals.items()))


if __name__ == "__main__":
    main()
