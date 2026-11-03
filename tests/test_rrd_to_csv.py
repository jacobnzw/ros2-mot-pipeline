import csv
import tempfile
import unittest
from pathlib import Path

from scripts.rrd_to_csv import (
    ROS2_TIMESTAMP_COLUMN,
    STEADY_ELAPSED_COLUMN,
    Sample,
    select_timestamp_column,
    stage_from_entity_path,
    timestamp_to_unix_ms,
    write_csv,
)


class RrdToCsvTests(unittest.TestCase):
    def test_stage_from_entity_path_selects_latency_duration_and_intervals(self) -> None:
        self.assertEqual(stage_from_entity_path("/detector/inference_latency_ms"), "detector_inference")
        self.assertEqual(stage_from_entity_path("/detector/callback_duration_ms"), "detector_callback_duration")
        self.assertEqual(stage_from_entity_path("/tracker/callback_interval_ms"), "tracker_callback_interval")
        self.assertIsNone(stage_from_entity_path("/detector/frame_drop_count"))
        self.assertIsNone(stage_from_entity_path("/tracker/active_tracks"))

    def test_export_prefers_steady_elapsed_and_falls_back_to_ros_timeline(self) -> None:
        self.assertEqual(
            select_timestamp_column([ROS2_TIMESTAMP_COLUMN, STEADY_ELAPSED_COLUMN]),
            STEADY_ELAPSED_COLUMN,
        )
        self.assertEqual(select_timestamp_column([ROS2_TIMESTAMP_COLUMN]), ROS2_TIMESTAMP_COLUMN)
        with self.assertRaisesRegex(ValueError, "no supported time timeline"):
            select_timestamp_column(["log_time"])

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
            Sample(timestamp_ns=100_000_000, stage="detector_inference", value=8.42),
            Sample(timestamp_ns=200_000_000, stage="pipeline_end_to_end", value=24.61),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "latency.csv"
            write_csv(samples, output_path, "run-001", now_ms=1_000_000)

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(
            rows,
            [
                ["999900", "run-001", "detector_inference", "8.42"],
                ["1000000", "run-001", "pipeline_end_to_end", "24.61"],
            ],
        )

    def test_csv_escapes_run_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "latency.csv"
            write_csv([Sample(1_000_000_000, "detector_inference", 1.0)], output_path, "run,one")

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows[0][1], "run,one")


if __name__ == "__main__":
    unittest.main()
