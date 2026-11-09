#!/usr/bin/env python3

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

ARRAY_SUMMARY_METRICS = {
    "HOTA": "HOTA___AUC",
    "DetA": "DetA___AUC",
    "AssA": "AssA___AUC",
    "DetRe": "DetRe___AUC",
    "DetPr": "DetPr___AUC",
    "AssRe": "AssRe___AUC",
    "AssPr": "AssPr___AUC",
    "LocA": "LocA___AUC",
    "OWTA": "OWTA___AUC",
}
SCALAR_SUMMARY_METRICS = (
    "HOTA(0)",
    "LocA(0)",
    "HOTALocA(0)",
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
    "sMOTA",
    "IDF1",
    "IDR",
    "IDP",
    "IDTP",
    "IDFN",
    "IDFP",
    "Dets",
    "GT_Dets",
    "IDs",
    "GT_IDs",
)


@dataclass(frozen=True)
class KpiSample:
    tracker: str
    sequence: str
    class_name: str
    metric: str
    value: float


def _read_detailed_file(path: Path) -> list[KpiSample]:
    tracker = path.parent.name
    class_name = path.name.removesuffix("_detailed.csv")
    samples = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "seq" not in reader.fieldnames:
            raise ValueError(f"TrackEval detailed file has no 'seq' column: {path}")

        columns = {
            **ARRAY_SUMMARY_METRICS,
            **{metric: metric for metric in SCALAR_SUMMARY_METRICS},
        }
        missing_columns = set(columns.values()) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"TrackEval detailed file {path} is missing columns: {', '.join(sorted(missing_columns))}")

        for row in reader:
            sequence = row["seq"]
            if not sequence:
                raise ValueError(f"TrackEval detailed file contains a row without a sequence: {path}")
            for metric, column in columns.items():
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid value for {metric} in {path}, sequence {sequence}") from error
                if not math.isfinite(value):
                    raise ValueError(f"non-finite value for {metric} in {path}, sequence {sequence}")
                samples.append(KpiSample(tracker, sequence, class_name, metric, value))
    return samples


def read_trackeval_samples(results_folder: Path) -> tuple[list[KpiSample], list[Path]]:
    detailed_files = sorted(results_folder.rglob("*_detailed.csv"))
    if not detailed_files:
        raise ValueError(f"no TrackEval *_detailed.csv files found under {results_folder}")

    samples = [sample for path in detailed_files for sample in _read_detailed_file(path)]
    return samples, detailed_files


def write_kpi_csv(
    samples: list[KpiSample],
    output_path: Path,
    run_id: str,
    timestamp_ms: int,
    hardware: str = "",
    quantization: str = "",
    label: str = "",
) -> None:
    if not samples:
        raise ValueError("TrackEval results contain no KPI samples")
    if not run_id:
        raise ValueError("run label must not be empty")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        for sample in samples:
            writer.writerow(
                (
                    timestamp_ms,
                    run_id,
                    hardware,
                    quantization,
                    label,
                    sample.tracker,
                    sample.sequence,
                    sample.class_name,
                    sample.metric,
                    sample.value,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TrackEval detailed results as VictoriaMetrics KPI CSV rows.")
    parser.add_argument(
        "results_folder",
        type=Path,
        help="Folder containing TrackEval tracker subfolders and *_detailed.csv files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: trackeval.csv in the results folder)",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID (default: results folder name)",
    )
    parser.add_argument(
        "--timestamp-ms",
        type=int,
        help="Evaluation timestamp in Unix milliseconds (default: newest detailed file modification time)",
    )
    parser.add_argument("--hw", default="", help="Hardware label")
    parser.add_argument("--quant", default="", help="Quantization label")
    parser.add_argument("--label", default="", help="Additional user-defined label")
    args = parser.parse_args()

    results_folder = args.results_folder.expanduser()
    if not results_folder.is_dir():
        parser.error(f"TrackEval results folder does not exist: {results_folder}")

    output_path = args.output.expanduser() if args.output else results_folder / "trackeval.csv"
    run_id = args.run_id if args.run_id is not None else results_folder.name
    if not run_id:
        parser.error("run label must not be empty")

    try:
        samples, detailed_files = read_trackeval_samples(results_folder)
        timestamp_ms = (
            args.timestamp_ms
            if args.timestamp_ms is not None
            else max(path.stat().st_mtime_ns for path in detailed_files) // 1_000_000
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_kpi_csv(
            samples,
            output_path,
            run_id,
            timestamp_ms,
            hardware=args.hw,
            quantization=args.quant,
            label=args.label,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Exported {len(samples):,} TrackEval KPI samples to {output_path}")
    print(f"Run label: {run_id}")
    print(f"Hardware: {args.hw}; quantization: {args.quant}; label: {args.label}")
    print(f"Evaluation timestamp: {timestamp_ms} ms since Unix epoch")


if __name__ == "__main__":
    main()
