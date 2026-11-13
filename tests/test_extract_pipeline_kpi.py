import csv
import tempfile
import unittest
from pathlib import Path

from scripts.eval.extract_pipeline_kpi import (
    ROS2_TIMESTAMP_COLUMN,
    STEADY_ELAPSED_COLUMN,
    Sample,
    select_timestamp_column,
    name_from_entity_path,
    timestamp_to_unix_ms,
    write_csv,
)


class RrdToCsvTests(unittest.TestCase):
    def test_name_from_entity_path_preserves_millisecond_units(self) -> None:
        self.assertEqual(name_from_entity_path("/detector/inference_latency_ms"), "detector_inference_ms")
        self.assertEqual(name_from_entity_path("/detector/callback_duration_ms"), "detector_callback_duration_ms")
        self.assertEqual(name_from_entity_path("/detector/callback_interval_ms"), "detector_callback_interval_ms")
        self.assertEqual(name_from_entity_path("/tracker/callback_interval_ms"), "tracker_callback_interval_ms")
        self.assertEqual(name_from_entity_path("/tracker/association_latency_ms"), "tracker_association_ms")
        self.assertEqual(name_from_entity_path("/tracker/callback_duration_ms"), "tracker_callback_duration_ms")
        self.assertEqual(name_from_entity_path("/pipeline/end_to_end_latency_ms"), "pipeline_end_to_end_ms")
        self.assertIsNone(name_from_entity_path("/detector/frame_drop_count"))
        self.assertIsNone(name_from_entity_path("/tracker/active_tracks"))

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
            Sample(timestamp_ns=100_000_000, name="detector_inference_ms", value=8.42),
            Sample(timestamp_ns=200_000_000, name="pipeline_end_to_end_ms", value=24.61),
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

        self.assertEqual(
            rows,
            [
                [
                    "999900",
                    "run-001",
                    "detector_inference_ms",
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
                    "pipeline_end_to_end_ms",
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
                [Sample(1_000_000_000, "detector_inference_ms", 1.0)],
                output_path,
                "run,one",
                hardware="Jetson, Orin",
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(rows[0][1], "run,one")
        self.assertEqual(rows[0][3], "Jetson, Orin")
        self.assertEqual(len(rows[0]), 10)

    def test_config_defaults_to_empty_csv_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "latency.csv"
            write_csv(
                [Sample(1_000_000_000, "detector_inference_ms", 1.0)],
                output_path,
                "run-001",
                now_ms=1_000_000,
            )

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.reader(csv_file))

        self.assertEqual(row, ["1000000", "run-001", "detector_inference_ms", "", "", "", "1.0", "", "", "1.0"])


if __name__ == "__main__":
    unittest.main()
