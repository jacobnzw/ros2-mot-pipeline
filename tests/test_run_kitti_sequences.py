import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import yaml

from scripts.eval.run_kitti_sequences import (
    count_mcap_frames,
    create_trackeval_gt,
    detect_hardware,
    discover_sequences,
    identify_hardware,
    launch_command,
    main,
    parse_sequences,
    positive_float,
    start_monitor,
    write_metadata,
)


class RunKittiSequencesTests(unittest.TestCase):
    def test_identifies_jetson_and_gpu_names(self) -> None:
        self.assertEqual(identify_hardware("NVIDIA Jetson AGX Orin Developer Kit", None), "jetson")
        self.assertEqual(identify_hardware(None, "NVIDIA GeForce RTX 3090\n"), "rtx3090")
        self.assertEqual(identify_hardware(None, "NVIDIA GeForce RTX 4090\n"), "rtx4090")
        self.assertEqual(identify_hardware(None, "NVIDIA A100-SXM4-80GB\n"), "a100-sxm4-80gb")
        self.assertEqual(identify_hardware(None, "GeForce RTX 3090"), "rtx3090")
        with self.assertRaisesRegex(ValueError, "unidentified GPU"):
            identify_hardware(None, " \n")
        with self.assertRaisesRegex(ValueError, "unidentified GPU"):
            identify_hardware(None, "NVIDIA GeForce")
        with self.assertRaisesRegex(ValueError, "single GPU"):
            identify_hardware(None, "GPU 0\nGPU 1\n")

    def test_detects_gpu_zero_without_trailing_newline(self) -> None:
        with (
            patch("pathlib.Path.read_text", side_effect=OSError),
            patch("pathlib.Path.is_file", return_value=False),
            patch("scripts.eval.run_kitti_sequences.subprocess.run") as run,
        ):
            run.return_value = SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 4090\n")
            self.assertEqual(detect_hardware(), "rtx4090")
        self.assertIn("--id=0", run.call_args.args[0])

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

    def test_playback_rate_must_be_finite_and_positive(self) -> None:
        self.assertEqual(positive_float("1.5"), 1.5)
        for value in ("0", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_float(value)

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
                1.5,
                ["0020", "0001"],
            )

            self.assertEqual(
                yaml.safe_load(metadata.read_text(encoding="utf-8")),
                {
                    "n_threads": 4,
                    "hardware": "rtx3090",
                    "precision": "fp16",
                    "rate": 1.5,
                    "sequences": ["0020", "0001"],
                    "config": "nightly-run",
                    "config_dir": "runs/nightly-run",
                    "kitti_gt": str(kitti_gt),
                    "kitti_root": str(kitti_root),
                    "trackeval_gt": str(trackeval_gt.resolve()),
                    "trackeval_seqmap": str(trackeval_seqmap.resolve()),
                },
            )

    def test_builds_ros_launch_command_with_all_run_arguments(self) -> None:
        self.assertEqual(
            launch_command("0019", Path("/dataset/mcaps"), 4, "fp16", "nightly", 1.5, True),
            [
                "ros2",
                "launch",
                "launch/detector_test.launch.py",
                "n_threads:=4",
                "seq:=0019",
                "label:=nightly",
                "precision:=fp16",
                "kitti_root:=/dataset/mcaps",
                "rate:=1.5",
                "loop:=true",
            ],
        )

    def test_builds_ros_launch_command_without_loop_by_default(self) -> None:
        command = launch_command("0019", Path("/dataset/mcaps"), 1, "fp32", "baseline", 1.0)

        self.assertIn("loop:=false", command)

    def test_starts_the_unified_monitor_for_detected_hardware(self) -> None:
        for hardware in ("jetson", "rtx4090"):
            with self.subTest(hardware=hardware), patch("scripts.eval.run_kitti_sequences.subprocess.Popen") as popen:
                start_monitor(hardware, "nightly")
                self.assertEqual(
                    popen.call_args.args[0],
                    ["uv", "run", "scripts/monitor.py", "--hardware", hardware, "--config", "nightly"],
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
                1.0,
                ["0019"],
            )

            self.assertEqual(
                yaml.safe_load(metadata.read_text(encoding="utf-8"))["config"],
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

        with patch("scripts.eval.run_kitti_sequences.Reader", return_value=reader):
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
                "scripts.eval.run_kitti_sequences.count_mcap_frames",
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


class RepeatSequenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        module = "scripts.eval.run_kitti_sequences"
        self.sequences = [Path("/dataset/0001"), Path("/dataset/0019")]
        self.discover = self.stack.enter_context(patch(f"{module}.discover_sequences", return_value=self.sequences))
        self.stack.enter_context(patch(f"{module}.detect_hardware", return_value="jetson"))
        self.stack.enter_context(patch(f"{module}.create_trackeval_gt", return_value=Path("/eval/seqmap")))
        self.stack.enter_context(patch(f"{module}.write_metadata"))
        self.monitor = Mock()
        self.start_monitor = self.stack.enter_context(patch(f"{module}.start_monitor", return_value=self.monitor))
        self.stop_monitor = self.stack.enter_context(patch(f"{module}.stop_monitor"))
        self.launch_run = self.stack.enter_context(
            patch(f"{module}.subprocess.run", return_value=SimpleNamespace(returncode=0))
        )
        self.stack.enter_context(patch("builtins.print"))
        self.arguments = [
            "run_kitti_sequences.py",
            "--kitti-root",
            "/dataset",
            "--n-threads",
            "4",
            "--precision",
            "fp16",
            "--config",
            "test",
        ]

    def test_runs_each_sequence_in_order_for_default_and_repeat_count(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        commands = [
            launch_command(sequence.name, Path("/dataset"), 4, "fp16", "test", 1.0) for sequence in self.sequences
        ]

        for repeat_arguments, count in (([], 1), (["--repeat", "1"], 1), (["--repeat", "3"], 3)):
            with self.subTest(repeat_arguments=repeat_arguments):
                self.launch_run.reset_mock()
                self.stop_monitor.reset_mock()
                with patch("sys.argv", self.arguments + repeat_arguments):
                    self.assertEqual(main(), 0)
                self.assertEqual(
                    self.launch_run.call_args_list,
                    [call(command, cwd=repository_root, check=False) for command in commands for _ in range(count)],
                )
                self.stop_monitor.assert_called_once_with(self.monitor)

    def test_rejects_invalid_repeat_counts_before_starting(self) -> None:
        for value in ("0", "-1", "1.5", "invalid"):
            with self.subTest(value=value), patch("sys.argv", self.arguments + ["--repeat", value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
        self.discover.assert_not_called()
        self.start_monitor.assert_not_called()
        self.launch_run.assert_not_called()

    def test_failed_repeat_stops_remaining_runs_and_monitor(self) -> None:
        self.launch_run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=7),
        ]
        with patch("sys.argv", self.arguments + ["--repeat", "3"]):
            self.assertEqual(main(), 7)
        self.assertEqual(self.launch_run.call_count, 4)
        self.assertEqual(
            [
                next(arg for arg in invocation.args[0] if arg.startswith("seq:="))
                for invocation in self.launch_run.call_args_list
            ],
            ["seq:=0001", "seq:=0001", "seq:=0001", "seq:=0019"],
        )
        self.stop_monitor.assert_called_once_with(self.monitor)


if __name__ == "__main__":
    unittest.main()
