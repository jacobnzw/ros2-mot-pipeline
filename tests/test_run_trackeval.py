import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.eval.run_trackeval import (
    discover_tracker_sequences,
    parse_sequences,
    read_sequence_entries,
    run_trackeval,
)


def read_sequence_entry(seqmap_path: Path, sequence: str) -> str:
    return read_sequence_entries(seqmap_path, [sequence])[0]


class MotTrackEvalTests(unittest.TestCase):
    def test_selects_sequence_entry_from_whitespace_seqmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seqmap = Path(directory) / "evaluate_tracking.seqmap.training"
            seqmap.write_text(
                "0018 /data/image_02/0018 000000 000837\n0019 /data/image_02/0019 000000 001059\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_sequence_entry(seqmap, "0019"),
                "0019 /data/image_02/0019 000000 001059",
            )

    def test_missing_sequence_reports_seqmap_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seqmap = Path(directory) / "evaluate_tracking.seqmap.training"
            seqmap.write_text("0019 /data/image_02/0019 000000 001059\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sequence '0018' is not listed"):
                read_sequence_entry(seqmap, "0018")

    def test_duplicate_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seqmap = Path(directory) / "evaluate_tracking.seqmap.training"
            seqmap.write_text(
                "0019 /data/image_02/0019 000000 001059\n0019 /data/image_02/other 000000 001059\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "appears more than once"):
                read_sequence_entry(seqmap, "0019")

    def test_selects_multiple_sequence_entries_in_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seqmap = Path(directory) / "evaluate_tracking.seqmap.training"
            seqmap.write_text(
                "0018 /data/image_02/0018 000000 000837\n"
                "0019 /data/image_02/0019 000000 001059\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_sequence_entries(seqmap, ["0019", "0018"]),
                [
                    "0019 /data/image_02/0019 000000 001059",
                    "0018 /data/image_02/0018 000000 000837",
                ],
            )

    def test_parses_comma_and_space_separated_sequence_lists(self) -> None:
        self.assertEqual(parse_sequences(["0018, 0019", "0020"]), ["0018", "0019", "0020"])
        self.assertIsNone(parse_sequences(None))

    def test_rejects_empty_or_duplicate_sequence_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_sequences(["0018,,0019"])
        with self.assertRaisesRegex(ValueError, "must not be repeated"):
            parse_sequences(["0018,0019", "0018"])

    def test_discovers_all_tracker_sequences_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_folder = Path(directory) / "data"
            data_folder.mkdir()
            (data_folder / "0019.txt").touch()
            (data_folder / "0018.txt").touch()
            (data_folder / "notes.csv").touch()
            self.assertEqual(discover_tracker_sequences(Path(directory)), ["0018", "0019"])

    def test_runs_trackeval_once_for_all_discovered_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_folder = root / "gt"
            gt_labels = gt_folder / "label_02"
            gt_labels.mkdir(parents=True)
            tracker_folder = root / "trackers" / "bytetrack"
            tracker_data = tracker_folder / "data"
            tracker_data.mkdir(parents=True)
            (gt_folder / "evaluate_tracking.seqmap.training").write_text(
                "0018 /data/image_02/0018 000000 000837\n"
                "0019 /data/image_02/0019 000000 001059\n",
                encoding="utf-8",
            )
            for sequence in ("0018", "0019"):
                (gt_labels / f"{sequence}.txt").touch()
                (tracker_data / f"{sequence}.txt").touch()

            captured_seqmap: list[str] = []
            dataset = Mock()
            dataset_factory = Mock(
                side_effect=lambda config: (
                    captured_seqmap.append(
                        (Path(config["GT_FOLDER"]) / "evaluate_tracking.seqmap.training").read_text(
                            encoding="utf-8"
                        )
                    )
                    or dataset
                )
            )
            evaluator = Mock()
            with (
                patch("scripts.eval.run_trackeval.trackeval.datasets.Kitti2DBox", dataset_factory),
                patch("scripts.eval.run_trackeval.trackeval.Evaluator", return_value=evaluator),
                patch("scripts.eval.run_trackeval.trackeval.metrics.HOTA", return_value="hota"),
                patch("scripts.eval.run_trackeval.trackeval.metrics.CLEAR", return_value="clear"),
                patch("scripts.eval.run_trackeval.trackeval.metrics.Identity", return_value="identity"),
            ):
                run_trackeval(
                    sequences=None,
                    gt_folder=gt_folder,
                    tracker_folder=tracker_folder,
                    output_folder=root / "output",
                    split="training",
                    classes=["car"],
                )

            self.assertEqual(
                captured_seqmap,
                [
                    "0018 /data/image_02/0018 000000 000837\n"
                    "0019 /data/image_02/0019 000000 001059\n"
                ],
            )
            evaluator.evaluate.assert_called_once_with([dataset], ["hota", "clear", "identity"])


if __name__ == "__main__":
    unittest.main()
