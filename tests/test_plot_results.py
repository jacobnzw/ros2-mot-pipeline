import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
from matplotlib.backends.backend_agg import FigureCanvasAgg

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


def inference_results() -> pd.DataFrame:
    frame = load_results()
    inference = frame[frame["metric"] == plot_results.LATENCY].copy()
    inference["metric"] = plot_results.INFERENCE_LATENCY
    inference["value"] += inference["rate"] + inference["n_threads"]
    inference.loc[(inference["precision"] == "fp16") | (inference["n_threads"] != 2), "value"] = 9999
    return pd.concat([frame, inference], ignore_index=True)


def timed_inference_results() -> pd.DataFrame:
    frame = inference_results()
    frame["timestamp_ms"] = 10000
    inference = frame["metric"] == plot_results.INFERENCE_LATENCY
    frame.loc[inference & (frame["value"] > 100), "timestamp_ms"] = 10750
    common = frame[inference & (frame["value"] < 10)]
    extra = frame[
        inference & (frame["hardware"] == "jetson") & (frame["precision"] == "fp32")
        & (frame["n_threads"] == 2) & (frame["rate"] == 1) & (frame["sequence"] == "0000")
    ].head(1).copy()
    extra["value"] = 104
    extra["timestamp_ms"] = 15000
    return pd.concat([frame, *[common] * 300, extra], ignore_index=True)


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

    def test_load_platform_histograms_use_matched_pooled_inference_samples(self) -> None:
        frame = inference_results()
        zero = ((frame["hardware"] == "jetson") & (frame["precision"] == "fp32")
                & (frame["n_threads"] == 2) & (frame["rate"] == 1)
                & (frame["metric"] == plot_results.INFERENCE_LATENCY))
        frame.loc[frame[zero].index[0], "value"] = 0
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_results.sns, "histplot", wraps=plot_results.sns.histplot) as histplot:
                paths = plot_results.plot_load_platform_effect(frame, Path(directory))
            self.assertEqual(
                {path.name for path in paths},
                {"load_platform_inference_latency.svg", "load_platform_inference_latency.png"},
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertEqual(histplot.call_count, 6)
            for index, call in enumerate(histplot.call_args_list):
                samples = call.kwargs["data"]
                self.assertEqual(set(samples["hardware"]), {("jetson", "rtx3090")[index // 3]})
                self.assertEqual(set(samples["rate"]), {plot_results.RATES[index % 3]})
                self.assertEqual(set(samples["precision"]), {"fp32"})
                self.assertEqual(set(samples["n_threads"]), {2})
                self.assertEqual(set(samples["sequence"]), {"0000", "0001"})
                self.assertEqual(set(samples["metric"]), {plot_results.INFERENCE_LATENCY})
                self.assertEqual(len(samples), 4)
                self.assertEqual(call.kwargs["stat"], "percent")
                self.assertEqual(call.kwargs["bins"], histplot.call_args_list[0].kwargs["bins"])
                self.assertEqual(call.kwargs["ax"].get_xscale(), "symlog")
                self.assertEqual(call.kwargs["ax"].get_yscale(), "log")
                self.assertEqual(call.kwargs["ax"].get_xlim()[0], 0)
                self.assertGreater(call.kwargs["ax"].get_xlim()[1], 105)
            bins = histplot.call_args_list[0].kwargs["bins"]
            self.assertEqual(bins[0], 0)
            self.assertEqual(bins[-1], 105)

    def test_load_platform_rejects_missing_config_or_samples(self) -> None:
        frame = inference_results()
        config = ((frame["hardware"] == "jetson") & (frame["precision"] == "fp32")
                  & (frame["n_threads"] == 2) & (frame["rate"] == 3))
        with self.assertRaisesRegex(ValueError, "jetson fp32 n_threads=2 rate=3.0: expected exactly one configuration"):
            plot_results.load_platform_data(frame[~config])
        missing = config & (frame["sequence"] == "0001") & (frame["metric"] == plot_results.INFERENCE_LATENCY)
        with self.assertRaisesRegex(ValueError, "incomplete detector_inference_latency_ms"):
            plot_results.load_platform_data(frame[~missing])
        with self.assertRaisesRegex(ValueError, "exactly two platforms"):
            plot_results.load_platform_data(frame[frame["hardware"] == "jetson"])

    def test_load_platform_mot_compares_rates_and_averages_classes(self) -> None:
        frame = load_results()
        hota = (frame["source"] == "mot") & (frame["sequence"] == "COMBINED") & (frame["metric"] == "HOTA")
        frame.loc[hota & (frame["class"] == "car"), "value"] = frame.loc[
            hota & (frame["class"] == "car"), "rate"
        ] * 0.1
        frame.loc[hota & (frame["class"] == "pedestrian"), "value"] = frame.loc[
            hota & (frame["class"] == "pedestrian"), "rate"
        ] * 0.2
        frame.loc[hota & ((frame["precision"] != "fp32") | (frame["n_threads"] != 2)), "value"] = 9999
        mot = plot_results.load_platform_mot_data(frame)
        scores = mot[(mot["hardware"] == "jetson") & (mot["metric"] == "HOTA")].set_index("rate")["value"]
        for rate, expected in (("1x", 15), ("2x", 30), ("3x", 45)):
            self.assertAlmostEqual(scores[rate], expected)
        counts = mot[(mot["hardware"] == "jetson") & (mot["metric"] == "IDSW")]["value"]
        self.assertEqual(list(counts), [15, 15, 15])
        self.assertEqual(len(mot), 30)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_results.sns, "barplot", wraps=plot_results.sns.barplot) as barplot:
                paths = plot_results.plot_load_platform_mot(frame, Path(directory))
            self.assertEqual({path.name for path in paths}, {"load_platform_mot.svg", "load_platform_mot.png"})
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertEqual(barplot.call_count, 4)
            for index, call in enumerate(barplot.call_args_list):
                samples = call.kwargs["data"]
                self.assertEqual(call.kwargs["x"], "metric")
                self.assertEqual(call.kwargs["hue"], "rate")
                self.assertEqual(call.kwargs["hue_order"], ("1x", "2x", "3x"))
                self.assertEqual(set(samples["rate"]), {"1x", "2x", "3x"})
                self.assertEqual(set(samples["hardware"]), {("jetson", "rtx3090")[index // 2]})
                self.assertEqual(set(samples["metric"]), set((plot_results.MOT_SCORES, plot_results.MOT_COUNTS)[index % 2]))
            self.assertEqual(len(barplot.call_args_list[0].kwargs["ax"].figure.axes), 4)

    def test_load_platform_mot_rejects_missing_class_and_sequence_inventory(self) -> None:
        frame = load_results()
        config = ((frame["hardware"] == "jetson") & (frame["precision"] == "fp32")
                  & (frame["n_threads"] == 2) & (frame["rate"] == 2))
        missing_class = (config & (frame["source"] == "mot") & (frame["sequence"] == "COMBINED")
                         & (frame["metric"] == "HOTA") & (frame["class"] == "car"))
        with self.assertRaisesRegex(ValueError, "one COMBINED HOTA per class"):
            plot_results.load_platform_mot_data(frame[~missing_class])
        missing_sequence = config & (frame["source"] == "sequence") & (frame["sequence"] == "0001")
        with self.assertRaisesRegex(ValueError, "sequence inventory differs"):
            plot_results.load_platform_mot_data(frame[~missing_sequence])

    def test_load_platform_boxplot_excludes_first_inference_and_summarizes_removed_samples(self) -> None:
        frame = inference_results()
        frame["timestamp_ms"] = 10000
        first_sequence = (frame["metric"] == plot_results.INFERENCE_LATENCY) & (frame["sequence"] == "0000")
        frame.loc[first_sequence, "timestamp_ms"] += frame.loc[first_sequence, "value"] * 100
        different_maximum = (
            (frame["metric"] == plot_results.INFERENCE_LATENCY) & (frame["sequence"] == "0001")
            & (frame["hardware"] == "rtx3090") & (frame["precision"] == "fp32")
            & (frame["n_threads"] == 2) & (frame["rate"] == 1)
        )
        frame.loc[different_maximum, "value"] = 113
        selected = (first_sequence & (frame["hardware"] == "jetson") & (frame["precision"] == "fp32")
                    & (frame["n_threads"] == 2) & (frame["rate"] == 1))
        later = frame[selected].head(1).copy()
        later["timestamp_ms"] = 20000
        later["value"] = 104
        common = [later.assign(timestamp_ms=11000 + index, value=6) for index in range(10)]
        frame = pd.concat([frame, later, *common], ignore_index=True)
        original_boxplot = plot_results.sns.boxplot

        def draw_boxplot(**kwargs):
            self.assertEqual(kwargs["ax"].get_yscale(), "linear")
            return original_boxplot(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_results.sns, "boxplot", side_effect=draw_boxplot) as boxplot:
                paths = plot_results.plot_load_platform_boxplot(frame, Path(directory))
            self.assertEqual(
                {path.name for path in paths},
                {"load_platform_inference_boxplot.svg", "load_platform_inference_boxplot.png"},
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            boxplot.assert_called_once()
            kwargs = boxplot.call_args.kwargs
            data = kwargs["data"]
            self.assertEqual((kwargs["x"], kwargs["y"], kwargs["hue"]), ("rate", "value", "hardware"))
            self.assertEqual(kwargs["order"], plot_results.RATES)
            self.assertEqual(kwargs["whis"], 1.5)
            self.assertEqual(set(data["precision"]), {"fp32"})
            self.assertEqual(set(data["n_threads"]), {2})
            self.assertEqual(set(data["metric"]), {plot_results.INFERENCE_LATENCY})
            self.assertEqual(len(data), 23)
            jetson = data[(data["hardware"] == "jetson") & (data["rate"] == 1)]
            self.assertEqual(set(jetson["value"]), {5, 6, 104})
            axis = kwargs["ax"]
            self.assertEqual(axis.get_yscale(), "symlog")
            self.assertEqual(axis.get_ylim()[0], 0)
            self.assertGreater(axis.get_ylim()[1], 104)
            self.assertEqual(axis.lines[4].get_ydata()[0], 6)
            self.assertIn(104, axis.lines[5].get_ydata())
            self.assertEqual(
                [text.get_text() for text in axis.texts],
                ["103.00", "113.00", "104.00", "104.00", "105.00", "105.00"],
            )
            for text, expected_x in zip(axis.texts, [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2], strict=True):
                self.assertAlmostEqual(text.get_position()[0], expected_x)
            canvas = FigureCanvasAgg(axis.figure)
            canvas.draw()
            renderer = canvas.get_renderer()
            maximum_y = axis.transData.transform((0, data["value"].max()))[1]
            for text in axis.texts:
                bounds = text.get_bbox_patch().get_window_extent(renderer)
                self.assertGreater(bounds.y0, maximum_y)
                self.assertLess(bounds.y1, axis.get_window_extent().y1)
            self.assertIn("n=2 per box", axis.figure.texts[0].get_text())

    def test_boxplot_shows_individual_counts_if_they_differ(self) -> None:
        data = plot_results.load_platform_data(inference_results())
        data["timestamp_ms"] = data["value"] * 100
        extra = data[(data["hardware"] == "rtx3090") & (data["rate"] == 1)].head(1).copy()
        extra["sequence"] = "0002"
        extra["run"] = "0002"
        data = pd.concat([data, extra], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_results, "load_platform_data", return_value=data):
                with patch.object(plot_results.sns, "boxplot", wraps=plot_results.sns.boxplot) as boxplot:
                    plot_results.plot_load_platform_boxplot(data, Path(directory))
            texts = [text.get_text() for text in boxplot.call_args.kwargs["ax"].texts]
            self.assertEqual(texts[1], "103.00\nn=3")
            self.assertTrue(all(text.endswith("n=2") for text in texts[:1] + texts[2:]))
            axis = boxplot.call_args.kwargs["ax"]
            canvas = FigureCanvasAgg(axis.figure)
            canvas.draw()
            renderer = canvas.get_renderer()
            maximum_y = axis.transData.transform((0, boxplot.call_args.kwargs["data"]["value"].max()))[1]
            for text in axis.texts:
                bounds = text.get_bbox_patch().get_window_extent(renderer)
                self.assertGreater(bounds.y0, maximum_y)
                self.assertLess(bounds.y1, axis.get_window_extent().y1)

    def test_first_inference_exclusion_rejects_ambiguous_or_missing_timestamps(self) -> None:
        data = plot_results.load_platform_data(inference_results())
        with self.assertRaisesRegex(ValueError, "timestamps must be present"):
            plot_results.exclude_first_inferences(data)
        data["timestamp_ms"] = 10000
        with self.assertRaisesRegex(ValueError, "multiple samples share the earliest timestamp"):
            plot_results.exclude_first_inferences(data)
        data["timestamp_ms"] += data["value"] * 100
        data.loc[data.index[0], "timestamp_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "timestamps must be present"):
            plot_results.exclude_first_inferences(data)
        data = data.dropna(subset=["timestamp_ms"])
        one_per_recording = data.sort_values("timestamp_ms").groupby(["config", "sequence", "run"]).head(1)
        with self.assertRaisesRegex(ValueError, "must retain at least one"):
            plot_results.exclude_first_inferences(one_per_recording)

    def test_extremes_use_recording_relative_time_and_first_inference(self) -> None:
        frame = timed_inference_results()
        data, extremes, limit = plot_results.load_platform_extremes(frame)
        self.assertEqual(limit, plot_results.load_platform_extreme_limit(data))
        self.assertLess(limit, 100)
        self.assertEqual(len(extremes), 7)
        self.assertEqual(extremes["first_inference"].sum(), 6)
        self.assertEqual(set(extremes["elapsed_seconds"]), {0.75, 5.0})
        late = extremes[extremes["elapsed_seconds"] == 5].iloc[0]
        self.assertEqual((late["hardware"], late["sequence"], late["first_inference"]), ("jetson", "0000", False))
        self.assertEqual(len(extremes[extremes["sequence"] == "0001"]), 6)

        missing_timestamp = frame.copy()
        selected = ((missing_timestamp["source"] == "pipeline") & (missing_timestamp["precision"] == "fp32")
                    & (missing_timestamp["n_threads"] == 2))
        index = missing_timestamp[selected].index[0]
        missing_timestamp.loc[index, "timestamp_ms"] = None
        with self.assertRaisesRegex(ValueError, "pipeline recording timestamps"):
            plot_results.load_platform_extremes(missing_timestamp)

    def test_extreme_figures_show_time_and_all_sequence_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(plot_results.sns, "histplot", wraps=plot_results.sns.histplot) as histplot:
                with patch.object(plot_results.sns, "barplot", wraps=plot_results.sns.barplot) as barplot:
                    paths = plot_results.plot_load_platform_extremes(timed_inference_results(), Path(directory))
            self.assertEqual(
                {path.name for path in paths},
                {f"load_platform_extreme_{aspect}.{extension}"
                 for aspect in ("times", "sequences") for extension in ("svg", "png")},
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertEqual(histplot.call_count, 6)
            self.assertEqual(barplot.call_count, 6)
            self.assertEqual(sum(len(call.kwargs["data"]) for call in histplot.call_args_list), 7)
            self.assertEqual(sum(sum(call.kwargs["y"]) for call in barplot.call_args_list), 7)
            for call in barplot.call_args_list:
                self.assertEqual(list(call.kwargs["x"]), ["0000", "0001"])
            self.assertEqual(histplot.call_args_list[0].kwargs["ax"].get_xscale(), "symlog")

    def test_extreme_data_allows_no_samples_above_cutoff(self) -> None:
        frame = timed_inference_results()
        frame.loc[frame["metric"] == plot_results.INFERENCE_LATENCY, "value"] = 5
        data, extremes, limit = plot_results.load_platform_extremes(frame)
        self.assertEqual(limit, 5)
        self.assertFalse(data.empty)
        self.assertTrue(extremes.empty)
        with tempfile.TemporaryDirectory() as directory:
            paths = plot_results.plot_load_platform_extremes(frame, Path(directory))
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_sequence_time_matches_recordings_and_uses_each_sequence_start(self) -> None:
        frame = timed_inference_results()
        data = plot_results.load_platform_data(frame)
        starts = {"0000": 10_000_000_000, "0001": 20_000_000_000}

        def read_samples(path: Path, export_end_ms: int) -> list[tuple[int, float, int]]:
            group = data[(data["config"] == path.parts[-3]) & (data["run"] == path.parts[-2])]
            sequence = path.parts[-2]
            return [
                (int(timestamp), value, starts[sequence] + (2_000_000_000 if value > 100 else 1_000_000_000))
                for timestamp, value in group[["timestamp_ms", "value"]].itertuples(index=False, name=None)
            ]

        with patch.object(plot_results, "first_image_timestamp", side_effect=lambda path: starts[path.stem]):
            with patch.object(plot_results, "recording_inference_timestamps", side_effect=read_samples):
                _, extremes, _ = plot_results.load_platform_sequence_extremes(frame, Path("runs"), Path("bags"))
                self.assertEqual(len(extremes), 7)
                self.assertEqual(set(extremes["elapsed_seconds"]), {2.0})
                self.assertEqual(set(extremes["rate"]), {1.0, 2.0, 3.0})
                with tempfile.TemporaryDirectory() as directory:
                    paths = plot_results.plot_load_platform_sequence_time(
                        frame, Path(directory), Path("runs"), Path("bags")
                    )
                    self.assertTrue(all(path.stat().st_size > 0 for path in paths))
                    self.assertEqual(
                        {path.name for path in paths},
                        {"load_platform_extreme_sequence_times.svg", "load_platform_extreme_sequence_times.png"},
                    )

        def mismatched_samples(path: Path, export_end_ms: int) -> list[tuple[int, float, int]]:
            samples = read_samples(path, export_end_ms)
            timestamp, value, source = samples[0]
            samples[0] = (timestamp, value + 1, source)
            return samples

        with patch.object(plot_results, "first_image_timestamp", return_value=0):
            with patch.object(plot_results, "recording_inference_timestamps", side_effect=mismatched_samples):
                with self.assertRaisesRegex(ValueError, "inference samples differ from Parquet"):
                    plot_results.load_platform_sequence_extremes(frame, Path("runs"), Path("bags"))
        with patch.object(plot_results, "first_image_timestamp", return_value=100_000_000_000):
            with patch.object(plot_results, "recording_inference_timestamps", side_effect=read_samples):
                with self.assertRaisesRegex(ValueError, "precedes the source sequence"):
                    plot_results.load_platform_sequence_extremes(frame, Path("runs"), Path("bags"))

        with patch.object(plot_results, "first_image_timestamp", return_value=0):
            with patch.object(plot_results, "recording_inference_timestamps", return_value=[]):
                with self.assertRaisesRegex(ValueError, "sample count differs"):
                    plot_results.load_platform_sequence_extremes(frame, Path("runs"), Path("bags"))

    def test_recording_reader_requires_source_timeline_and_reconstructs_export_time(self) -> None:
        def chunk(name: str, include_source: bool) -> SimpleNamespace:
            columns = {
                "steady_elapsed": pa.array([1_000_000_000, 2_000_000_000], type=pa.duration("ns")),
                plot_results.SCALAR_COMPONENT: pa.array([[3.0], [4.0]]),
            }
            if include_source:
                columns["ros2_timestamp"] = pa.array([100_000_000, 200_000_000], type=pa.timestamp("ns"))
            batch = pa.record_batch(columns)
            return SimpleNamespace(entity_path=name, to_record_batch=lambda: batch)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.rrd"
            path.touch()
            reader = SimpleNamespace(stream=lambda: None)
            stream = SimpleNamespace(filter=lambda **kwargs: [chunk("/detector/inference_latency_ms", True)])
            with patch.object(plot_results.rerun_bindings, "RrdReaderInternal", return_value=reader):
                with patch.object(plot_results.rr.chunk, "LazyChunkStream", return_value=stream):
                    self.assertEqual(
                        plot_results.recording_inference_timestamps(path, 10_000),
                        [(9_000, 3.0, 100_000_000), (10_000, 4.0, 200_000_000)],
                    )
                missing = SimpleNamespace(filter=lambda **kwargs: [chunk("/detector/inference_latency_ms", False)])
                with patch.object(plot_results.rr.chunk, "LazyChunkStream", return_value=missing):
                    with self.assertRaisesRegex(ValueError, "lacks the ros2_timestamp timeline"):
                        plot_results.recording_inference_timestamps(path, 10_000)
            with self.assertRaisesRegex(FileNotFoundError, "source sequence does not exist"):
                plot_results.first_image_timestamp(Path(directory) / "missing.mcap")
            with self.assertRaisesRegex(FileNotFoundError, "recording does not exist"):
                plot_results.recording_inference_timestamps(Path(directory) / "missing.rrd", 10_000)


if __name__ == "__main__":
    unittest.main()
