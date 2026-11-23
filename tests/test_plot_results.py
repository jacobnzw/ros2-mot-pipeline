import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import plot_results


def example_results() -> pd.DataFrame:
    rows = []
    for hardware in ("jetson", "rtx3090"):
        for precision in ("fp32", "fp16"):
            base = {
                "hardware": hardware, "precision": precision, "n_threads": 2, "rate": 1.0,
                "config": f"{hardware}-{precision}", "tracker": None, "class": None,
            }
            for sequence in ("0000", "0001"):
                rows.append({**base, "source": "sequence", "sequence": sequence, "metric": "input_frame_count",
                             "value": 10, "run": None})
                values = [1, 2, 3] if sequence == "0000" else [100]
                for value in values:
                    rows.append({**base, "source": "pipeline", "sequence": sequence,
                                 "metric": plot_results.LATENCY, "value": value, "run": sequence})
                for value in (0, 3 if sequence == "0000" else 7):
                    rows.append({**base, "source": "pipeline", "sequence": sequence,
                                 "metric": plot_results.DROPS, "value": value, "run": sequence})
            for object_class, factor in (("car", 1), ("pedestrian", 2)):
                for metric in (*plot_results.MOT_SCORES, *plot_results.MOT_COUNTS):
                    value = 0.3 * factor if metric in plot_results.MOT_SCORES else 10 * factor
                    rows.append({**base, "source": "mot", "sequence": "COMBINED", "tracker": "bytetrack",
                                 "class": object_class, "metric": metric, "value": value, "run": None})
                    rows.append({**base, "source": "mot", "sequence": "0000", "tracker": "bytetrack",
                                 "class": object_class, "metric": metric, "value": 999, "run": None})
    return pd.DataFrame(rows)


def threading_results() -> pd.DataFrame:
    frames = []
    for threads in plot_results.THREAD_COUNTS:
        frame = example_results()
        frame["n_threads"] = threads
        frame["config"] += f"-t{threads}"
        frame.loc[(frame["source"] == "pipeline") & (frame["metric"] == plot_results.LATENCY), "value"] += threads
        frames.append(frame)
    other_rate = frames[0].copy()
    other_rate["rate"] = 2.0
    other_rate["value"] = 9999
    return pd.concat([*frames, other_rate], ignore_index=True)


def load_results() -> pd.DataFrame:
    frames = []
    for threads in plot_results.THREAD_COUNTS:
        for rate in plot_results.RATES:
            frame = example_results()
            frame["n_threads"] = threads
            frame["rate"] = rate
            frame["config"] += f"-t{threads}-r{rate}"
            drops = (frame["source"] == "pipeline") & (frame["metric"] == plot_results.DROPS)
            frame.loc[drops, "value"] = frame.loc[drops, "value"] * rate + threads
            frame.loc[drops & (frame["precision"] == "fp16"), "value"] = 9999
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class PlotResultsTests(unittest.TestCase):
    def test_pools_latency_sums_recording_maxima_and_averages_combined_classes(self) -> None:
        latency, drops, mot = plot_results.precision_effect_data(example_results())
        for frame in (latency, drops, mot):
            self.assertEqual(set(frame["hardware"]), {"jetson", "rtx3090"})
            self.assertEqual(set(frame["precision"]), {"fp16", "fp32"})
        self.assertTrue(latency.loc[latency["metric"] == "p50", "value"].eq(2.5).all())
        self.assertTrue(latency.loc[latency["metric"] == "p95", "value"].sub(85.45).abs().lt(1e-8).all())
        self.assertTrue(drops["value"].eq(10).all())
        self.assertTrue(mot.loc[mot["metric"] == "HOTA", "value"].sub(45).abs().lt(1e-8).all())
        self.assertTrue(mot.loc[mot["metric"] == "IDSW", "value"].eq(15).all())

    def test_rejects_missing_sequences_and_classes(self) -> None:
        frame = example_results()
        missing = frame[~((frame["hardware"] == "jetson") & (frame["precision"] == "fp16")
                          & (frame["sequence"] == "0001") & (frame["metric"] == plot_results.LATENCY))]
        with self.assertRaisesRegex(ValueError, "incomplete pipeline_end_to_end_latency_ms"):
            plot_results.precision_effect_data(missing)
        missing = frame[~((frame["hardware"] == "jetson") & (frame["precision"] == "fp16")
                          & (frame["sequence"] == "COMBINED") & (frame["class"] == "pedestrian")
                          & (frame["metric"] == "HOTA"))]
        with self.assertRaisesRegex(ValueError, "one COMBINED HOTA per class"):
            plot_results.precision_effect_data(missing)

    def test_writes_three_figures_in_both_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = plot_results.plot_precision_effect(example_results(), Path(directory))
            self.assertEqual(
                {path.name for path in paths},
                {f"precision_effect_{aspect}.{extension}"
                 for aspect in ("latency", "drops", "mot") for extension in ("svg", "png")},
            )
            for path in paths:
                self.assertGreater(path.stat().st_size, 0)

    def test_platform_figures_compare_platforms_within_each_precision(self) -> None:
        frame = example_results()
        frame.loc[(frame["hardware"] == "jetson") & (frame["precision"] == "fp16")
                  & (frame["source"] == "pipeline") & (frame["metric"] == plot_results.LATENCY), "value"] += 10
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_results.sns, "barplot", wraps=plot_results.sns.barplot) as barplot:
                paths = plot_results.plot_platform_effect(frame, Path(directory))
            self.assertEqual(
                {path.name for path in paths},
                {f"platform_effect_{aspect}.{extension}"
                 for aspect in ("latency", "drops", "mot") for extension in ("svg", "png")},
            )
            for path in paths:
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(barplot.call_count, 8)
            for call in barplot.call_args_list:
                self.assertEqual(call.kwargs["hue"], "hardware")
                self.assertEqual(call.kwargs["x"], "metric")
                self.assertEqual(set(call.kwargs["data"]["hardware"]), {"jetson", "rtx3090"})
                self.assertEqual(call.kwargs["data"]["precision"].nunique(), 1)
            fp16_latency = barplot.call_args_list[1].kwargs["data"]
            median = fp16_latency[fp16_latency["metric"] == "p50"].set_index("hardware")["value"]
            self.assertEqual(median["jetson"], 12.5)
            self.assertEqual(median["rtx3090"], 2.5)

    def test_platform_effect_requires_multiple_platforms(self) -> None:
        frame = example_results()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least two platforms"):
                plot_results.plot_platform_effect(frame[frame["hardware"] == "jetson"], Path(directory))

    def test_executor_plots_compare_threads_within_platform_at_fp32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_results.sns, "barplot", wraps=plot_results.sns.barplot) as barplot:
                paths = plot_results.plot_executor_effect(threading_results(), Path(directory))
            self.assertEqual(
                {path.name for path in paths},
                {f"executor_effect_{aspect}.{extension}"
                 for aspect in ("latency", "drops", "mot") for extension in ("svg", "png")},
            )
            for path in paths:
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(barplot.call_count, 8)
            for call in barplot.call_args_list:
                data = call.kwargs["data"]
                self.assertEqual(call.kwargs["x"], "metric")
                self.assertEqual(call.kwargs["hue"], "n_threads")
                self.assertEqual(call.kwargs["hue_order"], ("1", "2", "4"))
                self.assertEqual(set(data["n_threads"]), {"1", "2", "4"})
                self.assertEqual(data["hardware"].nunique(), 1)
                self.assertEqual(set(data["precision"]), {"fp32"})
            latency = barplot.call_args_list[0].kwargs["data"]
            median = latency[latency["metric"] == "p50"].set_index("n_threads")["value"]
            self.assertEqual(median.to_dict(), {"1": 3.5, "2": 4.5, "4": 6.5})

    def test_executor_rejects_missing_thread_configuration(self) -> None:
        frame = threading_results()
        frame = frame[~((frame["hardware"] == "jetson") & (frame["precision"] == "fp32")
                        & (frame["n_threads"] == 4))]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "jetson fp32 n_threads=4: expected exactly one configuration"):
                plot_results.plot_executor_effect(frame, Path(directory))

    def test_load_totals_sum_maxima_across_sequences_and_support_one_sequence(self) -> None:
        frame = load_results()
        totals = plot_results.load_effect_data(frame)
        selected = totals[(totals["hardware"] == "jetson") & (totals["n_threads"] == 4)]
        self.assertEqual(selected.set_index("rate")["value"].to_dict(), {1.0: 18, 2.0: 28, 3.0: 38})
        one = plot_results.load_effect_data(frame, "0000")
        selected = one[(one["hardware"] == "jetson") & (one["n_threads"] == 4)]
        self.assertEqual(selected.set_index("rate")["value"].to_dict(), {1.0: 7, 2.0: 10, 3.0: 13})

    def test_load_plots_rates_and_thread_lines_for_each_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_results.sns, "lineplot", wraps=plot_results.sns.lineplot) as lineplot:
                paths = plot_results.plot_load_effect(load_results(), Path(directory))
            self.assertEqual({path.name for path in paths}, {"load_effect_drops.svg", "load_effect_drops.png"})
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertEqual(lineplot.call_count, 2)
            for call in lineplot.call_args_list:
                data = call.kwargs["data"]
                self.assertEqual(call.kwargs["x"], "rate")
                self.assertEqual(call.kwargs["hue"], "n_threads")
                self.assertEqual(call.kwargs["hue_order"], ("1", "2", "4"))
                self.assertEqual(set(data["rate"]), {1.0, 2.0, 3.0})
                self.assertEqual(set(data["n_threads"]), {"1", "2", "4"})
                self.assertEqual(data["hardware"].nunique(), 1)
            selected_paths = plot_results.plot_load_effect(load_results(), Path(directory), "0000")
            self.assertEqual(
                {path.name for path in selected_paths}, {"load_effect_drops_0000.svg", "load_effect_drops_0000.png"}
            )

    def test_load_rejects_missing_configuration_or_samples(self) -> None:
        frame = load_results()
        config = (frame["hardware"] == "jetson") & (frame["precision"] == "fp32")
        config &= (frame["n_threads"] == 4) & (frame["rate"] == 3.0)
        with self.assertRaisesRegex(ValueError, "jetson fp32 n_threads=4 rate=3.0: expected exactly one configuration"):
            plot_results.load_effect_data(frame[~config])
        missing = config & (frame["sequence"] == "0001") & (frame["metric"] == plot_results.DROPS)
        with self.assertRaisesRegex(ValueError, "incomplete detector_frame_drop_count"):
            plot_results.load_effect_data(frame[~missing])
        with self.assertRaisesRegex(ValueError, "sequence '9999' is missing"):
            plot_results.load_effect_data(frame, "9999")
        with self.assertRaisesRegex(ValueError, "sequence must contain"):
            plot_results.load_effect_data(frame, "../outside")


if __name__ == "__main__":
    unittest.main()
