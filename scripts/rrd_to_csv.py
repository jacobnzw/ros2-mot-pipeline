#!/usr/bin/env python3

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import rerun_bindings

SCALAR_COMPONENT = "Scalars:scalars"
ROS2_TIMESTAMP_COLUMN = "ros2_timestamp"
STEADY_ELAPSED_COLUMN = "steady_elapsed"
EARLIEST_EXPECTED_UNIX_TIMESTAMP_MS = 946_684_800_000


@dataclass(frozen=True)
class Sample:
    timestamp_ns: int
    stage: str
    value: float


def stage_from_entity_path(entity_path: str) -> str | None:
    stage = entity_path.strip("/").replace("/", "_")
    if stage.endswith("_latency_ms"):
        return stage[: -len("_latency_ms")]
    for suffix in ("_duration_ms", "_interval_ms"):
        if stage.endswith(suffix):
            return stage[: -len("_ms")]
    return None


def select_timestamp_column(column_names: list[str]) -> str:
    if STEADY_ELAPSED_COLUMN in column_names:
        return STEADY_ELAPSED_COLUMN
    if ROS2_TIMESTAMP_COLUMN in column_names:
        return ROS2_TIMESTAMP_COLUMN
    raise ValueError("scalar chunk has no supported time timeline")


def read_latency_samples(recording_path: Path) -> list[Sample]:
    reader = rerun_bindings.RrdReaderInternal(str(recording_path))
    stream = rr.chunk.LazyChunkStream(reader.stream()).filter(components=SCALAR_COMPONENT)
    samples: list[Sample] = []

    for chunk in stream:
        stage = stage_from_entity_path(chunk.entity_path)
        if stage is None:
            continue

        batch = chunk.to_record_batch()
        timeline_column = select_timestamp_column(batch.schema.names)
        timestamps = batch.column(timeline_column)
        values = batch.column(SCALAR_COMPONENT)
        for row_index in range(batch.num_rows):
            timestamp = timestamps[row_index]
            row_values = values[row_index].as_py()
            if not timestamp.is_valid or row_values is None:
                continue

            timestamp_ns = timestamp.value
            samples.extend(
                Sample(timestamp_ns, stage, float(value))
                for value in row_values
                if value is not None and math.isfinite(float(value))
            )

    return samples


def timestamp_to_unix_ms(timestamp_ns: int, min_timestamp_ns: int, max_timestamp_ns: int, now_ms: int) -> int:
    if min_timestamp_ns >= EARLIEST_EXPECTED_UNIX_TIMESTAMP_MS * 1_000_000:
        return timestamp_ns // 1_000_000
    return now_ms - (max_timestamp_ns - timestamp_ns) // 1_000_000


def write_csv(
    samples: list[Sample],
    output_path: Path,
    run_id: str,
    now_ms: int | None = None,
    hardware: str = "",
    quantization: str = "",
    sequence: str = "",
    label: str = "",
) -> None:
    if not samples:
        raise ValueError("recording contains no finite latency or duration scalar samples")
    if not run_id:
        raise ValueError("run label must not be empty")

    min_timestamp_ns = min(sample.timestamp_ns for sample in samples)
    max_timestamp_ns = max(sample.timestamp_ns for sample in samples)
    export_time_ms = time.time_ns() // 1_000_000 if now_ms is None else now_ms

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        for sample in samples:
            writer.writerow(
                (
                    timestamp_to_unix_ms(sample.timestamp_ns, min_timestamp_ns, max_timestamp_ns, export_time_ms),
                    run_id,
                    sample.stage,
                    hardware,
                    quantization,
                    sequence,
                    label,
                    sample.value,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Rerun latency and duration scalars as VictoriaMetrics CSV import rows."
    )
    parser.add_argument("recording", type=Path, help="Input Rerun .rrd file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: latency.csv beside the recording)",
    )
    parser.add_argument(
        "--run-id",
        help="Run label (default: recording directory name, or recording filename stem)",
    )
    parser.add_argument("--hw", default="", help="Hardware label")
    parser.add_argument("--quant", default="", help="Quantization label")
    parser.add_argument("--seq", default="", help="Sequence label")
    parser.add_argument("--label", default="", help="Additional user-defined label")
    args = parser.parse_args()

    recording_path = args.recording.expanduser()
    if not recording_path.is_file():
        parser.error(f"recording file does not exist: {recording_path}")
    if recording_path.suffix.lower() != ".rrd":
        parser.error(f"expected a .rrd recording: {recording_path}")

    output_path = (
        args.output.expanduser() if args.output else recording_path.with_name("latency.csv")
    )
    if output_path.resolve() == recording_path.resolve():
        parser.error("output CSV must not overwrite the input recording")

    run_id = args.run_id
    if run_id is None:
        run_id = recording_path.parent.name if recording_path.stem == "recording" else recording_path.stem

    try:
        samples = read_latency_samples(recording_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(
            samples,
            output_path,
            run_id,
            hardware=args.hw,
            quantization=args.quant,
            sequence=args.seq,
            label=args.label,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Exported {len(samples):,} latency/duration samples to {output_path}")
    print(f"Run label: {run_id}")
    print(f"Hardware: {args.hw}; quantization: {args.quant}; sequence: {args.seq}; label: {args.label}")


if __name__ == "__main__":
    main()
