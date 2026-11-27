import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import plot_sequence_variability as plot


def example_data() -> pd.DataFrame:
    base = {"config": "variance", "sequence": "0019"}
    rows = [
        {**base, "source": "sequence", "metric": "input_frame_count", "value": 4, "run": None},
        {**base, "source": "pipeline", "metric": plot.DETECTOR_DROPS, "value": 0, "run": "run-c"},
        {**base, "source": "pipeline", "metric": plot.LATENCY, "value": 10, "run": "run-a", "ros2_timestamp_ns": 0},
        {**base, "source": "pipeline", "metric": plot.LATENCY, "value": 20, "run": "run-b", "ros2_timestamp_ns": 0},
        {**base, "source": "pipeline", "metric": plot.LATENCY, "value": 30, "run": "run-a",
         "ros2_timestamp_ns": 100_000_000},
        {**base, "source": "pipeline", "metric": plot.DETECTOR_DROPS, "value": 0, "run": "run-a"},
        {**base, "source": "pipeline", "metric": plot.DETECTOR_DROPS, "value": 2, "run": "run-a"},
        {**base, "source": "pipeline", "metric": plot.DETECTOR_DROPS, "value": 3, "run": "run-b"},
    ]
    frame = pd.DataFrame(rows)
    frame["ros2_timestamp_ns"] = frame["ros2_timestamp_ns"].astype("Int64")
    return frame


class SequenceVariabilityTests(unittest.TestCase):
    def test_uses_one_maximum_cumulative_drop_count_per_run(self) -> None:
        latency, timeline, drop_totals, runs = plot.variability_data(example_data(), "variance", "0019", 10)
        self.assertEqual(runs, 3)
        self.assertEqual(list(timeline["ros2_timestamp_ns"]), [0, 100_000_000, 200_000_000, 300_000_000])
        self.assertEqual(
            dict(zip(drop_totals["run"], drop_totals["total_dropped_frames"], strict=True)),
            {"run-a": 2, "run-b": 3, "run-c": 0},
        )

    def test_requires_drop_counter_samples_for_every_run(self) -> None:
        frame = example_data().drop(index=7)
        with self.assertRaisesRegex(ValueError, "missing detector_frame_drop_count samples.*run-b"):
            plot.variability_data(frame, "variance", "0019", 10)

    def test_rejects_missing_invalid_or_duplicate_source_timestamps(self) -> None:
        frame = example_data()
        with self.assertRaisesRegex(ValueError, "lacks ros2_timestamp_ns"):
            plot.variability_data(frame.drop(columns="ros2_timestamp_ns"), "variance", "0019", 10)
        frame.loc[2, "ros2_timestamp_ns"] = pd.NA
        with self.assertRaisesRegex(ValueError, "lack source timestamps"):
            plot.variability_data(frame, "variance", "0019", 10)
        frame.loc[2, "ros2_timestamp_ns"] = 1
        with self.assertRaisesRegex(ValueError, "do not match"):
            plot.variability_data(frame, "variance", "0019", 10)
        frame.loc[2, "ros2_timestamp_ns"] = 100_000_000
        with self.assertRaisesRegex(ValueError, "duplicate source timestamp"):
            plot.variability_data(frame, "variance", "0019", 10)

    def test_stacks_sorted_samples_and_leaves_missing_frames_empty(self) -> None:
        latency, timeline, _, runs = plot.variability_data(example_data(), "variance", "0019", 10)
        matrix = plot.stacked_latency_matrix(latency.iloc[::-1], timeline, runs)
        np.testing.assert_equal(
            matrix,
            [[10, 30, np.nan, np.nan], [20, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
        )

    def test_plots_single_frame_and_single_run(self) -> None:
        frame = example_data().loc[[0, 3, 7]].copy()
        frame.loc[0, "value"] = 1
        with tempfile.TemporaryDirectory() as directory:
            paths = plot.plot_variability(frame, "variance", "0019", 10, Path(directory))
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_generates_plots_for_other_frame_rates_and_extreme_latencies(self) -> None:
        frame = example_data()
        frame.loc[4, "ros2_timestamp_ns"] = 500_000_000
        frame.loc[2, "value"] = 2500
        figures = []
        original_subplots = plot.plt.subplots

        def capture_figure(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            figures.append(figure)
            return figure, axes

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot.plt, "subplots", side_effect=capture_figure):
                paths = plot.plot_variability(frame, "variance", "0019", 2, Path(directory))
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertEqual(
                {path.name for path in paths},
                {
                    "variance_latency_variability.svg",
                    "variance_latency_variability.png",
                    "variance_total_dropped_frames.svg",
                    "variance_total_dropped_frames.png",
                },
            )
        histogram = figures[1].axes[0]
        self.assertEqual(sum(patch.get_height() for patch in histogram.patches), 3)
        main, detail = figures[0].axes[:2]
        self.assertEqual(main.images[0].get_extent(), [-0.25, 1.75, 0.5, 3.5])
        self.assertEqual(detail.get_xlim(), (-0.25, 1.75))
        self.assertEqual(main.images[0].norm.vmax, 2500)
        self.assertEqual(main.get_ylabel(), "Sample rank")


if __name__ == "__main__":
    unittest.main()
