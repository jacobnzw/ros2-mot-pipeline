#!/usr/bin/env python3

import argparse
import csv
import tempfile
from pathlib import Path

import trackeval


def parse_sequences(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None

    sequences = [sequence.strip() for value in values for sequence in value.split(",")]
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("sequence list must contain non-empty sequence numbers")
    if len(sequences) != len(set(sequences)):
        raise ValueError("sequence numbers must not be repeated")
    return sequences


def discover_tracker_sequences(tracker_folder: Path) -> list[str]:
    data_folder = tracker_folder / "data"
    if not data_folder.is_dir():
        raise ValueError(f"tracker data folder does not exist: {data_folder}")

    sequences = sorted(path.stem for path in data_folder.glob("*.txt") if path.is_file())
    if not sequences:
        raise ValueError(f"no tracker sequence .txt files found in {data_folder}")
    return sequences


def read_sequence_entries(seqmap_path: Path, sequences: list[str]) -> list[str]:
    if not seqmap_path.is_file():
        raise ValueError(f"TrackEval seqmap does not exist: {seqmap_path}")

    text = seqmap_path.read_text(encoding="utf-8")
    try:
        dialect = csv.Sniffer().sniff(text[:1024])
        rows = csv.reader(text.splitlines(), dialect)
    except csv.Error as error:
        raise ValueError(f"could not parse TrackEval seqmap: {seqmap_path}") from error

    rows_by_sequence: dict[str, list[list[str]]] = {}
    for row in rows:
        if row:
            rows_by_sequence.setdefault(row[0], []).append(row)

    entries = []
    for sequence in sequences:
        matches = rows_by_sequence.get(sequence, [])
        if not matches:
            raise ValueError(f"sequence {sequence!r} is not listed in {seqmap_path}")
        if len(matches) > 1:
            raise ValueError(f"sequence {sequence!r} appears more than once in {seqmap_path}")
        if len(matches[0]) < 4:
            raise ValueError(f"seqmap entry for {sequence!r} must have at least four columns")
        entries.append(" ".join(matches[0]))
    return entries


def validate_inputs(
    sequences: list[str] | None,
    gt_folder: Path,
    tracker_folder: Path,
    split: str,
) -> tuple[list[str], Path]:
    if not tracker_folder.is_dir():
        raise ValueError(f"tracker folder does not exist: {tracker_folder}")
    sequences = sequences if sequences is not None else discover_tracker_sequences(tracker_folder)
    if not sequences:
        raise ValueError("at least one sequence is required")

    tracker_data = tracker_folder / "data"
    gt_labels = gt_folder / "label_02"
    if not gt_labels.is_dir():
        raise ValueError(f"KITTI ground-truth label folder does not exist: {gt_labels}")

    missing_tracker_files = [
        tracker_data / f"{sequence}.txt" for sequence in sequences if not (tracker_data / f"{sequence}.txt").is_file()
    ]
    if missing_tracker_files:
        raise ValueError("tracker result file(s) do not exist: " + ", ".join(map(str, missing_tracker_files)))

    missing_gt_files = [
        gt_labels / f"{sequence}.txt" for sequence in sequences if not (gt_labels / f"{sequence}.txt").is_file()
    ]
    if missing_gt_files:
        raise ValueError("KITTI ground-truth label file(s) do not exist: " + ", ".join(map(str, missing_gt_files)))

    return sequences, gt_labels


def run_trackeval(
    sequences: list[str] | None,
    gt_folder: Path,
    tracker_folder: Path,
    output_folder: Path,
    split: str,
    classes: list[str],
) -> None:
    sequences, gt_labels = validate_inputs(
        sequences=sequences,
        gt_folder=gt_folder,
        tracker_folder=tracker_folder,
        split=split,
    )

    seqmap_name = f"evaluate_tracking.seqmap.{split}"
    seqmap_entries = read_sequence_entries(gt_folder / seqmap_name, sequences)

    output_folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mot-trackeval-") as temp_dir:
        sequence_gt_folder = Path(temp_dir)
        (sequence_gt_folder / "label_02").symlink_to(gt_labels.resolve(), target_is_directory=True)
        (sequence_gt_folder / seqmap_name).write_text("\n".join(seqmap_entries) + "\n", encoding="utf-8")

        dataset_config = {
            "GT_FOLDER": str(sequence_gt_folder),
            "TRACKERS_FOLDER": str(tracker_folder.parent.resolve()),
            "TRACKERS_TO_EVAL": [tracker_folder.name],
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_FOLDER": str(output_folder.resolve()),
            "SPLIT_TO_EVAL": split,
            "CLASSES_TO_EVAL": classes,
        }
        eval_config = {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": True,
            "PRINT_CONFIG": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            "LOG_ON_ERROR": None,
        }

        dataset = trackeval.datasets.Kitti2DBox(dataset_config)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
        evaluator = trackeval.Evaluator(eval_config)
        evaluator.evaluate([dataset], metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI tracking sequences with TrackEval (car and pedestrian classes)."
    )
    parser.add_argument(
        "--sequence",
        nargs="+",
        help="Sequence IDs to evaluate (space- or comma-separated; default: all tracker data/<sequence>.txt files)",
    )
    parser.add_argument(
        "--gt-folder",
        required=True,
        type=Path,
        help="KITTI split directory containing label_02/ and evaluate_tracking.seqmap.<split>",
    )
    parser.add_argument(
        "--tracker-folder",
        required=True,
        type=Path,
        help="Tracker-specific directory containing data/<sequence>.txt",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        type=Path,
        help="Directory for TrackEval summaries and detailed results",
    )
    parser.add_argument(
        "--split",
        default="training",
        help="KITTI split name used in the seqmap filename (default: training)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=["car", "pedestrian"],
        default=["car", "pedestrian"],
        help="KITTI classes to evaluate; TrackEval Kitti2DBox supports only car and pedestrian",
    )
    args = parser.parse_args()

    try:
        sequences = parse_sequences(args.sequence)
        run_trackeval(
            sequences=sequences,
            gt_folder=args.gt_folder.expanduser().resolve(),
            tracker_folder=args.tracker_folder.expanduser().resolve(),
            output_folder=args.output_folder.expanduser(),
            split=args.split,
            classes=args.classes,
        )
    except (OSError, RuntimeError, ValueError, trackeval.utils.TrackEvalException) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
