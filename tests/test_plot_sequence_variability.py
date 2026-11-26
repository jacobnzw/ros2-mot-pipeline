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
        {**base, "source": "pipeline", "metric": "detector_frame_drop_count", "value": 0, "run": "run-c"},
        {**base, "source": "pipeline", "metric": plot.LATENCY, "value": 10, "run": "run-a", "ros2_timestamp_ns": 0},
        {**base, "source": "pipeline", "metric": plot.LATENCY, "value": 20, "run": "run-b", "ros2_timestamp_ns": 0},
        {**base, "source": "pipeline", "metric": plot.LATENCY, "value": 30, "run": "run-a",
         "ros2_timestamp_ns": 100_000_000},
    ]
    frame = pd.DataFrame(rows)
    frame["ros2_timestamp_ns"] = frame["ros2_timestamp_ns"].astype("Int64")
    return frame


class SequenceVariabilityTests(unittest.TestCase):
    def test_counts_missing_samples_including_zero_observations_and_empty_run(self) -> None:
        latency, counts, runs = plot.variability_data(example_data(), "variance", "0019", 10)
        self.assertEqual(runs, 3)
        self.assertEqual(list(counts["observed_runs"]), [2, 1, 0, 0])
        self.assertEqual(list(counts["missing_runs"]), [1, 2, 3, 3])
        self.assertEqual(list(latency["observed_runs"]), [2, 2, 1])

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
        latency, counts, runs = plot.variability_data(example_data(), "variance", "0019", 10)
        matrix = plot.stacked_latency_matrix(latency.iloc[::-1], counts, runs)
        np.testing.assert_equal(
            matrix,
            [[10, 30, np.nan, np.nan], [20, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
        )

    def test_plots_single_frame_and_single_run(self) -> None:
        frame = example_data().iloc[[0, 2]].copy()
        frame.loc[0, "value"] = 1
        with tempfile.TemporaryDirectory() as directory:
            paths = plot.plot_variability(frame, "variance", "0019", 10, 0.2, Path(directory))
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
                paths = plot.plot_variability(frame, "variance", "0019", 2, 0.2, Path(directory))
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
        main, detail = figures[0].axes[:2]
        self.assertEqual(main.images[0].get_extent(), [-0.25, 1.75, 0.5, 3.5])
        self.assertEqual(detail.get_xlim(), (-0.25, 1.75))
        self.assertEqual(main.images[0].norm.vmax, 2500)
        self.assertEqual(main.get_ylabel(), "Sample rank")


if __name__ == "__main__":
    unittest.main()
