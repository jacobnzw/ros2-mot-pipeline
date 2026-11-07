#!/usr/bin/env python3

import argparse
import csv
import tempfile
from pathlib import Path

import trackeval


def read_sequence_entry(seqmap_path: Path, sequence: str) -> str:
    if not seqmap_path.is_file():
        raise ValueError(f"TrackEval seqmap does not exist: {seqmap_path}")

    text = seqmap_path.read_text(encoding="utf-8")
    try:
        dialect = csv.Sniffer().sniff(text[:1024])
        rows = csv.reader(text.splitlines(), dialect)
    except csv.Error as error:
        raise ValueError(f"could not parse TrackEval seqmap: {seqmap_path}") from error

    matches = [row for row in rows if row and row[0] == sequence]
    if not matches:
        raise ValueError(f"sequence {sequence!r} is not listed in {seqmap_path}")
    if len(matches) > 1:
        raise ValueError(f"sequence {sequence!r} appears more than once in {seqmap_path}")
    if len(matches[0]) < 4:
        raise ValueError(f"seqmap entry for {sequence!r} must have at least four columns")
    return " ".join(matches[0])


def run_trackeval(
    sequence: str,
    gt_folder: Path,
    tracker_folder: Path,
    output_folder: Path,
    split: str,
    classes: list[str],
) -> None:
    seqmap_name = f"evaluate_tracking.seqmap.{split}"
    seqmap_entry = read_sequence_entry(gt_folder / seqmap_name, sequence)
    gt_labels = gt_folder / "label_02"
    tracker_file = tracker_folder / "data" / f"{sequence}.txt"
    if not gt_labels.is_dir():
        raise ValueError(f"KITTI ground-truth label folder does not exist: {gt_labels}")
    if not (gt_labels / f"{sequence}.txt").is_file():
        raise ValueError(f"KITTI ground-truth labels do not exist for sequence {sequence}: {gt_labels}")
    if not tracker_file.is_file():
        raise ValueError(f"tracker result does not exist for sequence {sequence}: {tracker_file}")

    output_folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mot-trackeval-") as temp_dir:
        sequence_gt_folder = Path(temp_dir)
        (sequence_gt_folder / "label_02").symlink_to(gt_labels.resolve(), target_is_directory=True)
        (sequence_gt_folder / seqmap_name).write_text(seqmap_entry + "\n", encoding="utf-8")

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
        description="Evaluate one KITTI tracking sequence with TrackEval (car and pedestrian classes)."
    )
    parser.add_argument("--sequence", required=True, help="KITTI sequence ID, e.g. 0019")
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
        run_trackeval(
            sequence=args.sequence,
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
