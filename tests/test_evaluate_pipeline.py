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
                    "thread_num": 2,
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

    def test_config_selects_metadata_from_repository_runs_directory(self) -> None:
        results = {"pipeline_kpi_csvs": [], "tracker_kpi_csv": Path("tracker_kpi.csv")}
        with (
            patch("sys.argv", ["evaluate_pipeline.py", "--config", "baseline"]),
            patch.object(evaluate_pipeline, "_build_driver") as build_driver,
        ):
            build_driver.return_value.execute.return_value = results

            self.assertEqual(evaluate_pipeline._main(), 0)

        repository_root = Path(evaluate_pipeline.__file__).resolve().parents[2]
        build_driver.assert_called_once_with(4)
        build_driver.return_value.execute.assert_called_once_with(
            ["pipeline_kpi_csvs", "tracker_kpi_csv"],
            inputs={"metadata_path": repository_root / "runs" / "baseline" / "metadata.yaml"},
        )

    def test_metadata_resolves_run_and_ground_truth_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = self._create_run(Path(directory), ())
            metadata = evaluate_pipeline.run_metadata(metadata_path)

            self.assertEqual(metadata.config, "baseline")
            self.assertEqual(metadata.config_dir, metadata_path.parent)
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
            metadata_path = self._create_run(root, ("0019", "0018"))
            detail_file = root / "trackers" / "bytetrack" / "car_detailed.csv"
            detail_file.parent.mkdir(parents=True)
            detail_file.touch()
            active_readers = 0
            reader_lock = threading.Lock()
            readers_started_together = threading.Event()

            def read_samples(recording_path: Path) -> list[Sample]:
                nonlocal active_readers
                with reader_lock:
                    active_readers += 1
                    if active_readers == 2:
                        readers_started_together.set()
                if not readers_started_together.wait(timeout=5):
                    raise AssertionError("per-sequence work was not run concurrently")
                return [Sample(1_000_000_000, "detector_inference", 1.0)]

            def read_predictions(recording_path: Path, fps: float, first_frame_timestamp_ns: int) -> list:
                return []

            with (
                patch.object(evaluate_pipeline.extract_pipeline_kpi, "read_latency_samples", side_effect=read_samples),
                patch.object(
                    evaluate_pipeline.extract_tracker_preds_to_kitti,
                    "read_tracks",
                    side_effect=read_predictions,
                ),
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
            self.assertEqual(
                run_eval.call_args.kwargs["gt_folder"],
                metadata_path.parent / "eval",
            )
            self.assertEqual(len(results["pipeline_kpi_csvs"]), 2)
            self.assertTrue(all(path.is_file() for path in results["pipeline_kpi_csvs"]))
            self.assertTrue(results["tracker_kpi_csv"].is_file())
            with results["pipeline_kpi_csvs"][0].open(newline="", encoding="utf-8") as csv_file:
                self.assertEqual(next(csv.reader(csv_file))[7], "baseline")
            with results["tracker_kpi_csv"].open(newline="", encoding="utf-8") as csv_file:
                self.assertEqual(next(csv.reader(csv_file))[4], "baseline")
            self.assertTrue((metadata_path.parent / "eval" / "trackers" / "bytetrack" / "data" / "0018.txt").is_file())


if __name__ == "__main__":
    unittest.main()
