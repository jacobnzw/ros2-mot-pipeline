import csv
import tempfile
import unittest
from pathlib import Path

import rerun as rr

from scripts.eval.extract_pipeline_kpi import (
    ROS2_TIMESTAMP_COLUMN,
    STEADY_ELAPSED_COLUMN,
    Sample,
    read_latency_samples,
    select_timestamp_column,
    timestamp_to_unix_ms,
    write_csv,
)


class RrdToCsvTests(unittest.TestCase):
    def test_all_scalars_are_exported_with_normalized_entity_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording_path = Path(directory) / "recording.rrd"
            output_path = Path(directory) / "pipeline.csv"
            recording = rr.RecordingStream("test-frame-drop-export")
            recording.save(recording_path)
            for timestamp, detector_drops, tracker_drops in [(1.0, 0, 0), (2.0, 3, 5)]:
                recording.set_time("steady_elapsed", duration=timestamp)
                recording.log("detector/frame_drop_count", rr.Scalars(detector_drops))
                recording.log("tracker/frame_drop_count", rr.Scalars(tracker_drops))
                recording.log("tracker/active_tracks", rr.Scalars(2))
                recording.log("detector/inference_latency_ms", rr.Scalars(8.42))
            recording.disconnect()

            samples = read_latency_samples(recording_path)
            write_csv(samples, output_path, "run-001", now_ms=1_000_000, sequence="0019", config="baseline")

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))
            self.assertTrue(all(row[10] == "" and row[11] for row in rows))
            rows = [row[:10] for row in rows]

        self.assertCountEqual(
            rows,
            [
                ["999000", "run-001", "detector_frame_drop_count", "", "", "", "1.0", "0019", "baseline", "0.0"],
                ["999000", "run-001", "tracker_frame_drop_count", "", "", "", "1.0", "0019", "baseline", "0.0"],
                ["999000", "run-001", "tracker_active_tracks", "", "", "", "1.0", "0019", "baseline", "2.0"],
                [
                    "999000",
                    "run-001",
                    "detector_inference_latency_ms",
                    "",
                    "",
                    "",
                    "1.0",
                    "0019",
                    "baseline",
                    "8.42",
                ],
                ["1000000", "run-001", "detector_frame_drop_count", "", "", "", "1.0", "0019", "baseline", "3.0"],
                ["1000000", "run-001", "tracker_frame_drop_count", "", "", "", "1.0", "0019", "baseline", "5.0"],
                ["1000000", "run-001", "tracker_active_tracks", "", "", "", "1.0", "0019", "baseline", "2.0"],
                [
                    "1000000",
                    "run-001",
                    "detector_inference_latency_ms",
                    "",
                    "",
                    "",
                    "1.0",
                    "0019",
                    "baseline",
                    "8.42",
                ],
            ],
        )

    def test_export_prefers_steady_elapsed_and_falls_back_to_ros_timeline(self) -> None:
        self.assertEqual(
            select_timestamp_column([ROS2_TIMESTAMP_COLUMN, STEADY_ELAPSED_COLUMN]),
            STEADY_ELAPSED_COLUMN,
        )
        self.assertEqual(select_timestamp_column([ROS2_TIMESTAMP_COLUMN]), ROS2_TIMESTAMP_COLUMN)
        with self.assertRaisesRegex(ValueError, "no supported time timeline"):
            select_timestamp_column(["log_time"])

    def test_preserves_both_original_timelines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording_path = Path(directory) / "recording.rrd"
            output_path = Path(directory) / "pipeline.csv"
            recording = rr.RecordingStream("test-two-timelines")
            recording.save(recording_path)
            recording.set_time("steady_elapsed", duration=1.5)
            recording.set_time("ros2_timestamp", timestamp=0.1)
            recording.log("pipeline/end_to_end_latency_ms", rr.Scalars(42))
            recording.disconnect()

            samples = read_latency_samples(recording_path)
            self.assertEqual(samples, [Sample(1_500_000_000, "pipeline_end_to_end_latency_ms", 42.0,
                                              100_000_000, 1_500_000_000)])
            write_csv(samples, output_path, "run", now_ms=1_000_000)
            with output_path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.reader(csv_file))
            self.assertEqual(row[10:], ["100000000", "1500000000"])

    def test_relative_ros_time_is_anchored_to_export_time(self) -> None:
        self.assertEqual(timestamp_to_unix_ms(100_000_000, 100_000_000, 2_100_000_000, 1_000_000), 998_000)
        self.assertEqual(timestamp_to_unix_ms(2_100_000_000, 100_000_000, 2_100_000_000, 1_000_000), 1_000_000)

    def test_unix_ros_time_is_converted_directly_to_milliseconds(self) -> None:
        unix_timestamp_ns = 1_798_876_800_123_456_789
        self.assertEqual(
            timestamp_to_unix_ms(unix_timestamp_ns, unix_timestamp_ns, unix_timestamp_ns, 1_000_000),
            1_798_876_800_123,
        )

    def test_csv_rows_have_vm_import_column_order_and_no_header(self) -> None:
        samples = [
            Sample(timestamp_ns=100_000_000, name="detector_inference_latency_ms", value=8.42),
            Sample(timestamp_ns=200_000_000, name="pipeline_end_to_end_latency_ms", value=24.61),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "latency.csv"
            write_csv(
                samples,
                output_path,
                "run-001",
                hardware="jetson",
                precision="fp16",
                n_threads=4,
                rate=1.5,
                sequence="0020",
                config="baseline",
                now_ms=1_000_000,
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))
            self.assertTrue(all(row[10:] == ["", ""] for row in rows))
            rows = [row[:10] for row in rows]

        self.assertEqual(
            rows,
            [
                [
                    "999900",
                    "run-001",
                    "detector_inference_latency_ms",
                    "jetson",
                    "fp16",
                    "4",
                    "1.5",
                    "0020",
                    "baseline",
                    "8.42",
                ],
                [
                    "1000000",
                    "run-001",
                    "pipeline_end_to_end_latency_ms",
                    "jetson",
                    "fp16",
                    "4",
                    "1.5",
                    "0020",
                    "baseline",
                    "24.61",
                ],
            ],
        )

    def test_csv_escapes_run_id_and_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "latency.csv"
            write_csv(
                [Sample(1_000_000_000, "detector_inference_latency_ms", 1.0)],
                output_path,
                "run,one",
                hardware="Jetson, Orin",
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows[0][1], "run,one")
        self.assertEqual(rows[0][3], "Jetson, Orin")
        self.assertEqual(len(rows[0]), 12)

    def test_config_defaults_to_empty_csv_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "latency.csv"
            write_csv(
                [Sample(1_000_000_000, "detector_inference_latency_ms", 1.0)],
                output_path,
                "run-001",
                now_ms=1_000_000,
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.reader(csv_file))

        self.assertEqual(
            row, ["1000000", "run-001", "detector_inference_latency_ms", "", "", "", "1.0", "", "", "1.0", "", ""]
        )


if __name__ == "__main__":
    unittest.main()
