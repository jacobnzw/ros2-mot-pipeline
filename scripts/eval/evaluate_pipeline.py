#!/usr/bin/env python3

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from hamilton import driver, htypes
from hamilton.execution import executors

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval import extract_pipeline_kpi, extract_tracker_preds_to_kitti, run_trackeval, trackeval_to_vm


@dataclass(frozen=True)
class RunMetadata:
    config: str
    config_dir: Path
    hardware: str
    precision: str
    rate: float
    trackeval_gt: Path


@dataclass(frozen=True)
class RecordingJob:
    sequence: str
    recording_path: Path


@dataclass(frozen=True)
class SequenceOutput:
    sequence: str
    pipeline_kpi_csv: Path
    tracker_prediction_file: Path


def _resolve_metadata_path(value: str, metadata_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    repository_root = metadata_path.parent.parent.parent
    if metadata_path.parent.parent.name != "runs":
        repository_root = metadata_path.parent
    return (repository_root / path).resolve()


def _required_string(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata field {key!r} must be a non-empty string")
    return value


def _metadata_rate(metadata: dict[str, Any]) -> float:
    value = metadata.get("rate", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("metadata field 'rate' must be a finite number greater than zero")
    return float(value)


def run_metadata(metadata_path: Path) -> RunMetadata:
    metadata_path = metadata_path.expanduser().resolve()
    if not metadata_path.is_file():
        raise ValueError(f"run metadata file does not exist: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    if not isinstance(metadata, dict):
        raise ValueError(f"run metadata must contain a YAML mapping: {metadata_path}")

    config_key = "config" if "config" in metadata else "label"
    config = _required_string(metadata, config_key)
    config_dir_key = "config_dir" if "config_dir" in metadata else "label_dir"
    config_dir = _resolve_metadata_path(_required_string(metadata, config_dir_key), metadata_path)
    if not config_dir.is_dir():
        raise ValueError(f"run config directory does not exist: {config_dir}")

    hardware = _required_string(metadata, "hardware")
    precision = _required_string(metadata, "precision")
    rate = _metadata_rate(metadata)
    trackeval_gt = _resolve_metadata_path(_required_string(metadata, "trackeval_gt"), metadata_path)
    if not (trackeval_gt / "label_02").is_dir():
        raise ValueError(f"TrackEval ground-truth folder must contain label_02/: {trackeval_gt}")
    trackeval_seqmap = _resolve_metadata_path(_required_string(metadata, "trackeval_seqmap"), metadata_path)
    if not trackeval_seqmap.is_file():
        raise ValueError(f"TrackEval seqmap does not exist: {trackeval_seqmap}")
    if trackeval_seqmap.parent != trackeval_gt:
        raise ValueError(f"TrackEval seqmap must be inside the TrackEval ground-truth folder: {trackeval_seqmap}")

    return RunMetadata(
        config=config,
        config_dir=config_dir,
        hardware=hardware,
        precision=precision,
        rate=rate,
        trackeval_gt=trackeval_gt,
    )


def recording_job(run_metadata: RunMetadata) -> htypes.Parallelizable[RecordingJob]:
    recordings = sorted(run_metadata.config_dir.glob("*/recording.rrd"))
    if not recordings:
        raise ValueError(f"no recording.rrd files found directly under {run_metadata.config_dir}")

    jobs = []
    for recording_path in recordings:
        parts = recording_path.parent.name.split("_")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"recording directory must follow '<timestamp>_<sequence>': {recording_path.parent}")
        jobs.append(RecordingJob(sequence=parts[1], recording_path=recording_path))

    sequences = [job.sequence for job in jobs]
    duplicate_sequences = sorted({sequence for sequence in sequences if sequences.count(sequence) > 1})
    if duplicate_sequences:
        raise ValueError(f"multiple recordings found for sequence(s): {', '.join(duplicate_sequences)}")

    yield from sorted(jobs, key=lambda job: job.sequence)


def sequence_output(recording_job: RecordingJob, run_metadata: RunMetadata) -> SequenceOutput:
    pipeline_output_path = run_metadata.config_dir / "eval" / "pipeline" / f"kpi_{recording_job.sequence}.csv"
    pipeline_output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = extract_pipeline_kpi.read_latency_samples(recording_job.recording_path)
    extract_pipeline_kpi.write_csv(
        samples,
        pipeline_output_path,
        run_id=recording_job.recording_path.parent.name,
        hardware=run_metadata.hardware,
        precision=run_metadata.precision,
        rate=run_metadata.rate,
        sequence=recording_job.sequence,
        config=run_metadata.config,
    )

    tracker_output_path = (
        run_metadata.config_dir / "eval" / "trackers" / "bytetrack" / "data" / f"{recording_job.sequence}.txt"
    )
    tracker_output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks = extract_tracker_preds_to_kitti.read_tracks(
        recording_job.recording_path,
        fps=10.0,
        first_frame_timestamp_ns=0,
    )
    extract_tracker_preds_to_kitti.write_kitti_tracks(tracks, tracker_output_path)
    return SequenceOutput(
        sequence=recording_job.sequence,
        pipeline_kpi_csv=pipeline_output_path,
        tracker_prediction_file=tracker_output_path,
    )


# TODO: drop this? need?
def sequence_outputs(sequence_output: htypes.Collect[SequenceOutput]) -> list[SequenceOutput]:
    return sequence_output


def pipeline_kpi_csvs(sequence_outputs: list[SequenceOutput]) -> list[Path]:
    return [output.pipeline_kpi_csv for output in sequence_outputs]


def tracker_prediction_files(sequence_outputs: list[SequenceOutput]) -> list[Path]:
    return [output.tracker_prediction_file for output in sequence_outputs]


def trackeval_results_folder(tracker_prediction_files: list[Path], run_metadata: RunMetadata) -> Path:
    tracker_folder = run_metadata.config_dir / "eval" / "trackers" / "bytetrack"
    output_folder = run_metadata.config_dir / "eval" / "trackers"
    run_trackeval.run_trackeval(
        sequences=sorted(path.stem for path in tracker_prediction_files),
        gt_folder=run_metadata.trackeval_gt,
        tracker_folder=tracker_folder,
        output_folder=output_folder,
        split="training",
        classes=["car", "pedestrian"],
    )
    return output_folder


def tracker_kpi_csv(trackeval_results_folder: Path, run_metadata: RunMetadata) -> Path:
    samples, detailed_files = trackeval_to_vm.read_trackeval_samples(trackeval_results_folder)
    timestamp_ms = max(path.stat().st_mtime_ns for path in detailed_files) // 1_000_000
    output_path = run_metadata.config_dir / "eval" / "tracker_kpi.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trackeval_to_vm.write_kpi_csv(
        samples,
        output_path,
        timestamp_ms=timestamp_ms,
        hardware=run_metadata.hardware,
        precision=run_metadata.precision,
        rate=run_metadata.rate,
        config=run_metadata.config,
    )
    return output_path


def _build_driver(max_workers: int) -> driver.Driver:
    return (
        driver.Builder()
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_modules(sys.modules[__name__])
        .with_local_executor(executors.SynchronousLocalTaskExecutor())
        .with_remote_executor(executors.MultiThreadingExecutor(max_tasks=max_workers))
        .build()
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract pipeline and tracker KPIs for a detector run using an Apache Hamilton DAG."
    )
    metadata_group = parser.add_mutually_exclusive_group(required=True)
    metadata_group.add_argument("--metadata", type=Path, help="Path to the run metadata YAML")
    metadata_group.add_argument(
        "--config", help="Configuration name; loads runs/<config>/metadata.yaml from the repository root"
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=4,
        help="Maximum concurrent per-sequence extraction tasks (default: 2)",
    )
    args = parser.parse_args()

    if args.config is not None:
        if not args.config or args.config in {".", ".."} or "/" in args.config or "\\" in args.config:
            parser.error("--config must be a non-empty directory name without path separators")
        metadata_path = Path(__file__).resolve().parents[2] / "runs" / args.config / "metadata.yaml"
    else:
        metadata_path = args.metadata

    results = _build_driver(args.max_workers).execute(
        ["pipeline_kpi_csvs", "tracker_kpi_csv"],
        inputs={"metadata_path": metadata_path},
    )
    print("Pipeline KPI CSVs:")
    for path in results["pipeline_kpi_csvs"]:
        print(f"  {path}")
    print(f"Tracker KPI CSV: {results['tracker_kpi_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
