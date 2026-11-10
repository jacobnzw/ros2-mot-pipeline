import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from scripts.run_kitti_sequences import (
    count_mcap_frames,
    create_trackeval_gt,
    discover_sequences,
    identify_hardware,
    launch_command,
    parse_sequences,
    write_metadata,
)


class RunKittiSequencesTests(unittest.TestCase):
    def test_identifies_jetson_and_rtx3090(self) -> None:
        self.assertEqual(identify_hardware("NVIDIA Jetson AGX Orin Developer Kit", None), "jetson")
        self.assertEqual(identify_hardware(None, "NVIDIA GeForce RTX 3090"), "rtx3090")
        with self.assertRaisesRegex(ValueError, "unsupported or unidentified GPU"):
            identify_hardware(None, "NVIDIA GeForce RTX 4090")

    def test_discovers_sorted_sequences_and_requires_mcaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("0019", "0001"):
                sequence = root / name
                sequence.mkdir()
                (sequence / f"{name}.mcap").touch()

            self.assertEqual([path.name for path in discover_sequences(root)], ["0001", "0019"])

            (root / "0019" / "0019.mcap").unlink()
            with self.assertRaisesRegex(ValueError, "Missing MCAP"):
                discover_sequences(root)

    def test_selects_requested_sequences_in_argument_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("0019", "0001", "0020"):
                sequence = root / name
                sequence.mkdir()
                (sequence / f"{name}.mcap").touch()

            selected = discover_sequences(root, ["0020", "0001"])

        self.assertEqual([path.name for path in selected], ["0020", "0001"])

    def test_only_requires_mcaps_for_selected_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("0019", "0020"):
                (root / name).mkdir()
            (root / "0019" / "0019.mcap").touch()

            self.assertEqual([path.name for path in discover_sequences(root, ["0019"])], ["0019"])
            with self.assertRaisesRegex(ValueError, "Unknown sequence"):
                discover_sequences(root, ["9999"])

    def test_sequence_parser_rejects_empty_and_duplicate_entries(self) -> None:
        self.assertEqual(parse_sequences("0019, 0020"), ["0019", "0020"])
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "non-empty"):
            parse_sequences("0019,,0020")
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "must not be repeated"):
            parse_sequences("0019,0019")

    def test_writes_metadata_with_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.yaml"
            kitti_root = Path("/dataset/mcaps")
            kitti_gt = Path("/dataset/training/label_02")
            trackeval_gt = root / "runs" / "nightly-run" / "eval"
            trackeval_seqmap = trackeval_gt / "evaluate_tracking.seqmap.training"
            write_metadata(
                metadata,
                kitti_root,
                kitti_gt,
                trackeval_gt,
                trackeval_seqmap,
                4,
                "rtx3090",
                "fp16",
                "nightly-run",
            )

            self.assertEqual(
                yaml.safe_load(metadata.read_text(encoding="utf-8")),
                {
                    "thread_num": 4,
                    "hardware": "rtx3090",
                    "precision": "fp16",
                    "label": "nightly-run",
                    "label_dir": "runs/nightly-run",
                    "kitti_gt": str(kitti_gt),
                    "kitti_root": str(kitti_root),
                    "trackeval_gt": str(trackeval_gt.resolve()),
                    "trackeval_seqmap": str(trackeval_seqmap.resolve()),
                },
            )

    def test_builds_ros_launch_command_with_all_run_arguments(self) -> None:
        self.assertEqual(
            launch_command("0019", Path("/dataset/mcaps"), 4, "fp16", "nightly"),
            [
                "ros2",
                "launch",
                "launch/detector_test.launch.py",
                "thread_num:=4",
                "seq:=0019",
                "label:=nightly",
                "precision:=fp16",
                "kitti_root:=/dataset/mcaps",
            ],
        )

    def test_metadata_quotes_yaml_special_characters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.yaml"
            write_metadata(
                metadata,
                Path("/dataset/mcaps"),
                Path("/dataset/training/label_02"),
                Path("/runs/nightly/eval"),
                Path("/runs/nightly/eval/evaluate_tracking.seqmap.training"),
                1,
                "jetson",
                "fp16",
                'baseline: "nightly"',
            )

            self.assertEqual(
                yaml.safe_load(metadata.read_text(encoding="utf-8"))["label"],
                'baseline: "nightly"',
            )

    def test_counts_image_messages_in_mcap(self) -> None:
        reader = Mock()
        reader.__enter__ = Mock(return_value=reader)
        reader.__exit__ = Mock(return_value=False)
        reader.connections = [
            SimpleNamespace(topic="/camera/image_raw/compressed", msgtype="sensor_msgs/msg/CompressedImage"),
            SimpleNamespace(topic="/tf", msgtype="tf2_msgs/msg/TFMessage"),
        ]
        reader.messages.return_value = iter((object(), object(), object()))

        with patch("scripts.run_kitti_sequences.Reader", return_value=reader):
            self.assertEqual(count_mcap_frames(Path("/dataset/0001.mcap")), 3)

        reader.messages.assert_called_once_with(connections=[reader.connections[0]])

    def test_creates_run_specific_trackeval_gt_and_seqmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_dir = root / "dataset" / "label_02"
            label_dir.mkdir(parents=True)
            sequences = []
            for name in ("0001", "0019"):
                sequence_dir = root / "mcaps" / name
                sequence_dir.mkdir(parents=True)
                (sequence_dir / f"{name}.mcap").touch()
                sequences.append(sequence_dir)

            eval_folder = root / "runs" / "nightly" / "eval"
            with patch(
                "scripts.run_kitti_sequences.count_mcap_frames",
                side_effect=[447, 1059],
            ):
                seqmap_path = create_trackeval_gt(eval_folder, label_dir, sequences)

            self.assertEqual(seqmap_path, eval_folder / "evaluate_tracking.seqmap.training")
            self.assertTrue((eval_folder / "label_02").is_symlink())
            self.assertEqual((eval_folder / "label_02").resolve(), label_dir)
            self.assertEqual(
                seqmap_path.read_text(encoding="utf-8"),
                "0001 image_02/0001 000000 447\n"
                "0019 image_02/0019 000000 1059\n",
            )


if __name__ == "__main__":
    unittest.main()
