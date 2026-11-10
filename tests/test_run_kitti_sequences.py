import argparse
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.run_kitti_sequences import (
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
            metadata = Path(directory) / "metadata.yaml"
            write_metadata(metadata, 4, "rtx3090", "fp16", "nightly-run")

            self.assertEqual(
                yaml.safe_load(metadata.read_text(encoding="utf-8")),
                {
                    "thread_num": 4,
                    "hardware": "rtx3090",
                    "precision": "fp16",
                    "label": "nightly-run",
                    "label_dir": "runs/nightly-run",
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
            write_metadata(metadata, 1, "jetson", "fp16", 'baseline: "nightly"')

            self.assertEqual(
                yaml.safe_load(metadata.read_text(encoding="utf-8"))["label"],
                'baseline: "nightly"',
            )


if __name__ == "__main__":
    unittest.main()
