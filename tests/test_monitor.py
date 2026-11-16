import argparse
import unittest
from pathlib import Path

from scripts.monitor import FIELDS, build_parser, get_jetson_stats, make_sample_row, positive_float


class MonitorTests(unittest.TestCase):
    def test_parser_defaults_to_pc_with_common_output_name(self) -> None:
        args = build_parser().parse_args(["--config", "baseline"])

        self.assertEqual(args.hardware, "pc")
        self.assertEqual(args.out_name, "device_metrics")
        self.assertEqual(args.freq, 1.0)

    def test_frequency_must_be_finite_and_positive(self) -> None:
        self.assertEqual(positive_float("0.5"), 0.5)
        for value in ("0", "-1", "nan", "inf", "invalid"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_float(value)

    def test_jetson_stats_fit_the_shared_csv_schema(self) -> None:
        metrics = get_jetson_stats(
            {
                "cpu_temp": 42.0,
                "cpu_usage": 20.0,
                "gpu_temp": 40.0,
                "gpu_usage": 15.0,
                "emc_usage": 10.0,
                "ram_used": 2048,
                "ram_total": 8192,
                "swap_used": 0,
                "power": 12000,
                "fan_rpm": 900,
            }
        )

        self.assertEqual(set(metrics) | {"timestamp_ms", "config", "hardware"}, set(FIELDS))
        self.assertEqual(metrics["gpu_usage_pct"], 15.0)
        self.assertEqual(metrics["power_mw"], 12000)
        self.assertIsNone(metrics["gpu_mem_used_mb"])
        self.assertIsNone(metrics["aux_temp_c"])

    def test_sample_row_uses_nan_for_missing_metrics_and_includes_vm_labels(self) -> None:
        metrics: dict[str, float | str | None] = {field: None for field in FIELDS[1:-2]}
        metrics["cpu_temp_c"] = 42.0
        row = make_sample_row(
            "nightly",
            "jetson",
            metrics,
            timestamp_ms=1_798_876_800_000,
        )

        self.assertEqual(list(row), FIELDS)
        self.assertEqual(row["gpu_mem_used_mb"], "nan")
        self.assertEqual(row["cpu_temp_c"], 42.0)
        self.assertEqual(row["config"], "nightly")
        self.assertEqual(row["hardware"], "jetson")

    def test_victoria_metrics_format_matches_device_csv_columns(self) -> None:
        importer = Path(__file__).parents[1] / "scripts/eval/vm_grafana_ingest.sh"
        line = next(
            line for line in importer.read_text(encoding="utf-8").splitlines() if line.startswith("device_metrics_format=")
        )
        rules = [rule.split(":") for rule in line.split('"')[1].split(",")]
        expected = [["1", "time", "unix_ms"]]
        expected.extend(
            [str(index), "metric", field] for index, field in enumerate(FIELDS[1:14], start=2)
        )
        expected.extend([["15", "label", "config"], ["16", "label", "hardware"]])

        self.assertEqual(rules, expected)


if __name__ == "__main__":
    unittest.main()
