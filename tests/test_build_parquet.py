import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import yaml

from scripts import build_parquet


class BuildParquetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.config = self.runs / "example"
        self.pipeline = self.config / "eval" / "pipeline"
        self.pipeline.mkdir(parents=True)
        (self.config / "metadata.yaml").write_text(
            yaml.safe_dump(
                {
                    "config": "example",
                    "hardware": "rtx3090",
                    "precision": "fp32",
                    "n_threads": 2,
                    "rate": 1.0,
                    "sequences": ["0000", "0001"],
                }
            ),
            encoding="utf-8",
        )
        (self.config / "eval" / "evaluate_tracking.seqmap.training").write_text(
            "0000 image_02/0000 000000 154\n0001 image_02/0001 000000 447\n", encoding="utf-8"
        )
        for sequence in ("0000", "0001"):
            self.write_csv(
                self.pipeline / f"kpi_{sequence}.csv",
                [
                    [
                        1000,
                        f"run_{sequence}",
                        "detector_frame_drop_count",
                        "rtx3090",
                        "fp32",
                        2,
                        1.0,
                        sequence,
                        "example",
                        0,
                    ],
                    [
                        2000,
                        f"run_{sequence}",
                        "detector_frame_drop_count",
                        "rtx3090",
                        "fp32",
                        2,
                        1.0,
                        sequence,
                        "example",
                        3,
                    ],
                ],
            )
        self.write_csv(
            self.config / "eval" / "tracker_kpi.csv",
            [[3000, "rtx3090", "fp32", 2, 1.0, "example", "bytetrack", "COMBINED", "car", "HOTA", 0.5]],
        )
        device_row = [4000, *([1] * len(build_parquet.DEVICE_METRICS)), "example", "pc"]
        device_row[1] = "nan"
        self.write_csv(self.config / "eval" / "device_metrics.csv", [build_parquet.DEVICE_COLUMNS, device_row])
        self.output = self.root / "results.parquet"

    @staticmethod
    def write_csv(path: Path, rows: list[list]) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(rows)

    def test_builds_all_sources_in_batches_with_labels_and_nulls(self) -> None:
        with patch.object(build_parquet, "BATCH_SIZE", 1):
            counts = build_parquet.build_parquet(self.runs, "*", self.output)
        self.assertEqual(counts, {"pipeline": 4, "mot": 1, "hardware": 13, "sequence": 2})
        table = pq.read_table(self.output)
        self.assertEqual(table.schema, build_parquet.SCHEMA)
        self.assertNotIn("source_hardware", table.column_names)
        rows = table.to_pylist()
        self.assertEqual({row["hardware"] for row in rows}, {"rtx3090"})
        pipeline = [r for r in rows if r["source"] == "pipeline"]
        self.assertEqual([r["sequence"] for r in pipeline], ["0000", "0000", "0001", "0001"])
        self.assertEqual([r["value"] for r in pipeline], [0.0, 3.0, 0.0, 3.0])
        self.assertEqual(pipeline[0]["timestamp_ms"], 1000)
        mot = next(r for r in rows if r["source"] == "mot")
        self.assertEqual(
            (mot["sequence"], mot["class"], mot["tracker"], mot["value"]), ("COMBINED", "car", "bytetrack", 0.5)
        )
        hardware = next(r for r in rows if r["source"] == "hardware" and r["metric"] == "cpu_temp_c")
        self.assertIsNone(hardware["value"])
        self.assertIsNone(hardware["sequence"])
        lengths = {r["sequence"]: r["value"] for r in rows if r["source"] == "sequence"}
        self.assertEqual(lengths, {"0000": 154.0, "0001": 447.0})
        self.assertIsNone(next(r for r in rows if r["source"] == "sequence")["timestamp_ms"])
        self.assertTrue(all(r["ros2_timestamp_ns"] is None and r["steady_elapsed_ns"] is None for r in rows))

    def test_preserves_nanosecond_timelines_from_new_csv(self) -> None:
        self.write_csv(
            self.pipeline / "kpi_0000.csv",
            [
                [1000, "run_0000", "metric", "rtx3090", "fp32", 2, 1.0, "0000", "example", 1,
                 1_798_876_800_123_456_789, 1_234_567_890],
                [2000, "run_0000", "metric", "rtx3090", "fp32", 2, 1.0, "0000", "example", 2, "", ""],
            ],
        )
        build_parquet.build_parquet(self.runs, "*", self.output)
        rows = pq.read_table(self.output).to_pylist()
        sample = next(row for row in rows if row["source"] == "pipeline" and row["sequence"] == "0000")
        self.assertEqual(sample["ros2_timestamp_ns"], 1_798_876_800_123_456_789)
        self.assertEqual(sample["steady_elapsed_ns"], 1_234_567_890)
        missing = next(
            row
            for row in rows
            if row["source"] == "pipeline" and row["sequence"] == "0000" and row["timestamp_ms"] == 2000
        )
        self.assertIsNone(missing["ros2_timestamp_ns"])
        self.assertIsNone(missing["steady_elapsed_ns"])

    def test_incomplete_inputs_do_not_replace_existing_output(self) -> None:
        self.output.write_bytes(b"existing")
        (self.pipeline / "kpi_0001.csv").unlink()
        with self.assertRaisesRegex(ValueError, "pipeline CSV files do not match"):
            build_parquet.build_parquet(self.runs, "*", self.output)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(list(self.root.glob(".results.parquet.*.tmp")), [])

    def test_builds_repeated_runs_and_ignores_superseded_legacy_csv(self) -> None:
        self.write_csv(
            self.pipeline / "kpi_0001.csv",
            [[1000, "20261110T120000Z_0001", "metric", "rtx3090", "fp32", 2, 1.0, "0001", "example", 1]],
        )
        for timestamp, value in (("20261110T120000Z", 1), ("20261110T120002Z", 2)):
            run = f"{timestamp}_0001"
            self.write_csv(
                self.pipeline / f"kpi_{run}.csv",
                [[1000, run, "detector_frame_drop_count", "rtx3090", "fp32", 2, 1.0, "0001", "example", value]],
            )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            counts = build_parquet.build_parquet(self.runs, "*", self.output)

        self.assertIn("ignoring duplicate legacy", stderr.getvalue())
        self.assertEqual(counts["pipeline"], 4)
        rows = [row for row in pq.read_table(self.output).to_pylist() if row["source"] == "pipeline"]
        repeats = [row for row in rows if row["sequence"] == "0001"]
        self.assertEqual({row["run"] for row in repeats}, {"20261110T120000Z_0001", "20261110T120002Z_0001"})
        self.assertEqual({row["value"] for row in repeats}, {1.0, 2.0})

    def test_preserves_legacy_csv_for_a_distinct_run(self) -> None:
        self.write_csv(
            self.pipeline / "kpi_20261110T120000Z_0001.csv",
            [[1000, "20261110T120000Z_0001", "metric", "rtx3090", "fp32", 2, 1.0, "0001", "example", 1]],
        )
        build_parquet.build_parquet(self.runs, "*", self.output)
        rows = [row for row in pq.read_table(self.output).to_pylist() if row["source"] == "pipeline"]
        self.assertEqual(
            {row["run"] for row in rows if row["sequence"] == "0001"}, {"run_0001", "20261110T120000Z_0001"}
        )

    def test_rejects_per_run_csv_with_mismatched_run_id(self) -> None:
        self.write_csv(
            self.pipeline / "kpi_20261110T120000Z_0001.csv",
            [[1000, "wrong_0001", "metric", "rtx3090", "fp32", 2, 1.0, "0001", "example", 1]],
        )
        with self.assertRaisesRegex(ValueError, "run ID does not match"):
            build_parquet.build_parquet(self.runs, "*", self.output)

    def test_detected_gpu_name_is_preserved_across_sources(self) -> None:
        label = "rtx4090"
        metadata_path = self.config / "metadata.yaml"
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        metadata["hardware"] = label
        metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
        for path, hardware_column in (
            (self.pipeline / "kpi_0000.csv", 3),
            (self.pipeline / "kpi_0001.csv", 3),
            (self.config / "eval" / "tracker_kpi.csv", 1),
            (self.config / "eval" / "device_metrics.csv", -1),
        ):
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))
            for row in rows[1:] if path.name == "device_metrics.csv" else rows:
                row[hardware_column] = label
            self.write_csv(path, rows)

        build_parquet.build_parquet(self.runs, "*", self.output)
        rows = pq.read_table(self.output).to_pylist()
        self.assertEqual({row["source"] for row in rows}, {"pipeline", "mot", "hardware", "sequence"})
        self.assertEqual({row["hardware"] for row in rows}, {label})

    def test_rejects_mismatched_labels(self) -> None:
        self.write_csv(
            self.pipeline / "kpi_0000.csv",
            [[1000, "run_0000", "detector_frame_drop_count", "jetson", "fp32", 2, 1.0, "0000", "example", 0]],
        )
        with self.assertRaisesRegex(ValueError, "hardware does not match"):
            build_parquet.build_parquet(self.runs, "*", self.output)
        self.assertFalse(self.output.exists())

    def test_missing_hardware_is_reported(self) -> None:
        (self.config / "eval" / "device_metrics.csv").unlink()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            counts = build_parquet.build_parquet(self.runs, "*", self.output)
        self.assertEqual(counts["hardware"], 0)
        self.assertIn("hardware measurements are unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
