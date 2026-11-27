import csv
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.eval import evaluate_pipeline
from scripts.eval.extract_pipeline_kpi import Sample
from scripts.eval.trackeval_to_vm import KpiSample


class EvaluatePipelineTests(unittest.TestCase):
    def _create_run(self, root: Path, sequences: tuple[str, ...]) -> Path:
        run_dir = root / "runs" / "baseline"
        gt_folder = root / "dataset" / "training"
        label_dir = gt_folder / "label_02"
        label_dir.mkdir(parents=True)
        run_dir.mkdir(parents=True)
        eval_folder = run_dir / "eval"
        eval_folder.mkdir()
        (eval_folder / "label_02").symlink_to(label_dir)
        seqmap_path = eval_folder / "evaluate_tracking.seqmap.training"
        seqmap_path.write_text("0018 image_02/0018 000000 100\n0019 image_02/0019 000000 100\n", encoding="utf-8")
        for index, sequence in enumerate(sequences):
            recording_dir = run_dir / f"20261110T12000{index}Z_{sequence}"
            recording_dir.mkdir()
            (recording_dir / "recording.rrd").touch()

        metadata_path = run_dir / "metadata.yaml"
        metadata_path.write_text(
            yaml.safe_dump(
                {
                    "n_threads": 2,
                    "hardware": "rtx3090",
                    "precision": "fp16",
                    "rate": 1.5,
                    "config": "baseline",
                    "config_dir": "runs/baseline",
                    "trackeval_gt": str(eval_folder),
                    "trackeval_seqmap": str(seqmap_path),
                }
            ),
            encoding="utf-8",
        )
        return metadata_path

    def test_config_glob_evaluates_every_matching_run_and_reports_failures(self) -> None:
        results = {"pipeline_kpi_csvs": [], "tracker_kpi_csv": Path("tracker_kpi.csv")}
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory)
            for name in ("rtx_a", "rtx_b", "rtx_nometa", "orin_a"):
                (runs_dir / name).mkdir()
                if name != "rtx_nometa":
                    (runs_dir / name / "metadata.yaml").touch()
            (runs_dir / "rtx_file").touch()

            def execute(_outputs: list[str], inputs: dict) -> dict:
                if inputs["metadata_path"].parent.name == "rtx_b":
                    raise ValueError("boom")
                return results

            with (
                patch("sys.argv", ["evaluate_pipeline.py", "--config", "rtx*"]),
                patch.object(evaluate_pipeline, "RUNS_DIR", runs_dir),
                patch.object(evaluate_pipeline, "_build_driver") as build_driver,
            ):
                build_driver.return_value.execute.side_effect = execute

                self.assertEqual(evaluate_pipeline._main(), 1)

            build_driver.assert_called_once_with(4)
            evaluated = [
                call.kwargs["inputs"]["metadata_path"] for call in build_driver.return_value.execute.call_args_list
            ]
            self.assertEqual(evaluated, [runs_dir / "rtx_a" / "metadata.yaml", runs_dir / "rtx_b" / "metadata.yaml"])

    def test_config_single_name_evaluates_only_that_run(self) -> None:
        results = {"pipeline_kpi_csvs": [], "tracker_kpi_csv": Path("tracker_kpi.csv")}
        with tempfile.TemporaryDirectory() as directory:
            runs_dir = Path(directory)
            for name in ("baseline", "baseline_2"):
                (runs_dir / name).mkdir()
                (runs_dir / name / "metadata.yaml").touch()
            with (
                patch("sys.argv", ["evaluate_pipeline.py", "--config", "baseline"]),
                patch.object(evaluate_pipeline, "RUNS_DIR", runs_dir),
                patch.object(evaluate_pipeline, "_build_driver") as build_driver,
            ):
                build_driver.return_value.execute.return_value = results

                self.assertEqual(evaluate_pipeline._main(), 0)

            build_driver.return_value.execute.assert_called_once_with(
                ["pipeline_kpi_csvs", "tracker_kpi_csv"],
                inputs={"metadata_path": runs_dir / "baseline" / "metadata.yaml"},
            )

    def test_config_without_match_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("sys.argv", ["evaluate_pipeline.py", "--config", "nomatch*"]),
                patch.object(evaluate_pipeline, "RUNS_DIR", Path(directory)),
                self.assertRaises(SystemExit),
            ):
                evaluate_pipeline._main()

    def test_metadata_resolves_run_and_ground_truth_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = self._create_run(Path(directory), ())
            metadata = evaluate_pipeline.run_metadata(metadata_path)

            self.assertEqual(metadata.config, "baseline")
            self.assertEqual(metadata.config_dir, metadata_path.parent)
            self.assertEqual(metadata.n_threads, 2)
            self.assertEqual(metadata.rate, 1.5)
            self.assertEqual(metadata.trackeval_gt, metadata_path.parent / "eval")

    def test_metadata_still_reads_legacy_label_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = self._create_run(Path(directory), ())
            metadata_content = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            metadata_content["label"] = metadata_content.pop("config")
            metadata_content["label_dir"] = metadata_content.pop("config_dir")
            metadata_path.write_text(yaml.safe_dump(metadata_content), encoding="utf-8")

            metadata = evaluate_pipeline.run_metadata(metadata_path)

            self.assertEqual(metadata.config, "baseline")
            self.assertEqual(metadata.config_dir, metadata_path.parent)

    def test_metadata_defaults_missing_playback_rate_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = self._create_run(Path(directory), ())
            metadata_content = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            metadata_content.pop("rate")
            metadata_path.write_text(yaml.safe_dump(metadata_content), encoding="utf-8")

            metadata = evaluate_pipeline.run_metadata(metadata_path)

            self.assertEqual(metadata.rate, 1.0)

    def test_metadata_reads_legacy_thread_num(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = self._create_run(Path(directory), ())
            metadata_content = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            metadata_content["thread_num"] = metadata_content.pop("n_threads")
            metadata_path.write_text(yaml.safe_dump(metadata_content), encoding="utf-8")

            metadata = evaluate_pipeline.run_metadata(metadata_path)

            self.assertEqual(metadata.n_threads, 2)

    def test_metadata_rejects_invalid_playback_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = self._create_run(Path(directory), ())
            metadata_content = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            metadata_content["rate"] = 0
            metadata_path.write_text(yaml.safe_dump(metadata_content), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "rate.*finite number greater than zero"):
                evaluate_pipeline.run_metadata(metadata_path)

    def test_recording_jobs_discover_sequences_from_run_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = self._create_run(Path(directory), ("0019", "0018"))
            metadata = evaluate_pipeline.run_metadata(metadata_path)

            jobs = list(evaluate_pipeline.recording_job(metadata))

            self.assertEqual([job.sequence for job in jobs], ["0018", "0019"])
            self.assertTrue(all(job.recording_path.is_file() for job in jobs))

    def test_hamilton_runs_both_branches_and_evaluates_predictions_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = self._create_run(root, ("0019", "0018", "0019"))
            detail_file = root / "trackers" / "bytetrack" / "car_detailed.csv"
            detail_file.parent.mkdir(parents=True)
            detail_file.touch()
            active_readers = 0
            reader_lock = threading.Lock()
            readers_started_together = threading.Event()
            sample_values = {
                "20261110T120000Z_0019": 1.0,
                "20261110T120001Z_0018": 2.0,
                "20261110T120002Z_0019": 3.0,
            }

            def read_samples(recording_path: Path) -> list[Sample]:
                nonlocal active_readers
                with reader_lock:
                    active_readers += 1
                    if active_readers == 2:
                        readers_started_together.set()
                if not readers_started_together.wait(timeout=5):
                    raise AssertionError("per-sequence work was not run concurrently")
                return [
                    Sample(1_000_000_000, "detector_inference_latency_ms", sample_values[recording_path.parent.name])
                ]

            def read_predictions(recording_path: Path, fps: float, first_frame_timestamp_ns: int) -> list:
                return []

            with (
                patch.object(evaluate_pipeline.extract_pipeline_kpi, "read_latency_samples", side_effect=read_samples),
                patch.object(
                    evaluate_pipeline.extract_tracker_preds_to_kitti,
                    "read_tracks",
                    side_effect=read_predictions,
                ) as read_tracks,
                patch.object(evaluate_pipeline.run_trackeval, "run_trackeval") as run_eval,
                patch.object(
                    evaluate_pipeline.trackeval_to_vm,
                    "read_trackeval_samples",
                    return_value=(
                        [KpiSample("bytetrack", "0018", "car", "HOTA", 0.5)],
                        [detail_file],
                    ),
                ),
            ):
                results = evaluate_pipeline._build_driver(2).execute(
                    ["pipeline_kpi_csvs", "tracker_kpi_csv"],
                    inputs={"metadata_path": metadata_path},
                )

            run_eval.assert_called_once()
            self.assertEqual(run_eval.call_args.kwargs["sequences"], ["0018", "0019"])
            self.assertEqual(read_tracks.call_count, 2)
            self.assertEqual(
                {call.args[0].parent.name for call in read_tracks.call_args_list},
                {"20261110T120001Z_0018", "20261110T120002Z_0019"},
            )
            self.assertEqual(
                run_eval.call_args.kwargs["gt_folder"],
                metadata_path.parent / "eval",
            )
            pipeline_csvs = results["pipeline_kpi_csvs"]
            self.assertEqual(len(pipeline_csvs), 3)
            self.assertEqual(
                {path.name for path in pipeline_csvs},
                {"kpi_20261110T120000Z_0019.csv", "kpi_20261110T120001Z_0018.csv", "kpi_20261110T120002Z_0019.csv"},
            )
            for path in pipeline_csvs:
                with path.open(newline="", encoding="utf-8") as csv_file:
                    row = next(csv.reader(csv_file))
                    self.assertEqual(row[1], path.stem.removeprefix("kpi_"))
                    self.assertEqual(float(row[9]), sample_values[row[1]])
            self.assertTrue(results["tracker_kpi_csv"].is_file())
            with pipeline_csvs[0].open(newline="", encoding="utf-8") as csv_file:
                pipeline_row = next(csv.reader(csv_file))
                self.assertEqual(pipeline_row[5], "2")
                self.assertEqual(pipeline_row[8], "baseline")
            with results["tracker_kpi_csv"].open(newline="", encoding="utf-8") as csv_file:
                tracker_row = next(csv.reader(csv_file))
                self.assertEqual(tracker_row[3], "2")
                self.assertEqual(tracker_row[5], "baseline")
            self.assertTrue((metadata_path.parent / "eval" / "trackers" / "bytetrack" / "data" / "0018.txt").is_file())


if __name__ == "__main__":
    unittest.main()
