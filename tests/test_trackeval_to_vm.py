import csv
import tempfile
import unittest
from pathlib import Path

from scripts.eval.trackeval_to_vm import (
    KpiSample,
    read_trackeval_samples,
    write_kpi_csv,
)


class TrackEvalToCsvTests(unittest.TestCase):
    def test_reads_per_sequence_and_combined_summary_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracker_folder = Path(directory) / "bytetrack"
            tracker_folder.mkdir()
            detailed_file = tracker_folder / "car_detailed.csv"
            detailed_file.write_text(
                "seq,HOTA___AUC,DetA___AUC,AssA___AUC,DetRe___AUC,DetPr___AUC,AssRe___AUC,"
                "AssPr___AUC,LocA___AUC,OWTA___AUC,HOTA(0),LocA(0),HOTALocA(0),"
                "MOTA,MOTP,MODA,CLR_Re,CLR_Pr,MTR,PTR,MLR,CLR_TP,CLR_FN,CLR_FP,IDSW,MT,PT,ML,"
                "Frag,sMOTA,IDF1,IDR,IDP,IDTP,IDFN,IDFP,Dets,GT_Dets,IDs,GT_IDs\n"
                "0019,0.5,0.6,0.4,0.7,0.8,0.55,0.65,0.8,0.5,0.7,0.8,0.56,0.4,0.75,0.42,0.8,0.9,0.5,0.3,0.2,"
                "8,2,1,1,1,1,0,0.4,0.3,0.6,0.7,0.5,7,3,2,9,10,3,4\n"
                "COMBINED,0.5,0.6,0.4,0.7,0.8,0.55,0.65,0.8,0.5,0.7,0.8,0.56,0.4,0.75,0.42,0.8,0.9,0.5,0.3,0.2,"
                "8,2,1,1,1,1,0,0.4,0.3,0.6,0.7,0.5,7,3,2,9,10,3,4\n",
                encoding="utf-8",
            )

            samples, files = read_trackeval_samples(Path(directory))

        self.assertEqual(files, [detailed_file])
        self.assertEqual(len(samples), 2 * 39)
        self.assertIn(KpiSample("bytetrack", "0019", "car", "HOTA", 0.5), samples)
        self.assertIn(KpiSample("bytetrack", "COMBINED", "car", "IDF1", 0.6), samples)

    def test_writes_vm_import_csv_with_all_series_labels(self) -> None:
        sample = KpiSample("bytetrack", "0019", "car", "HOTA", 0.49423)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "trackeval.csv"
            write_kpi_csv(
                [sample],
                output_path,
                timestamp_ms=1_798_876_800_000,
                hardware="jetson-orin",
                precision="fp16",
                label="baseline",
            )
            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(
            rows,
            [
                [
                    "1798876800000",
                    "jetson-orin",
                    "fp16",
                    "baseline",
                    "bytetrack",
                    "0019",
                    "car",
                    "HOTA",
                    "0.49423",
                ]
            ],
        )

    def test_metadata_labels_default_to_empty_columns(self) -> None:
        sample = KpiSample("bytetrack", "0019", "car", "HOTA", 0.5)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "trackeval.csv"
            write_kpi_csv([sample], output_path, timestamp_ms=1_798_876_800_000)
            with output_path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.reader(csv_file))

        self.assertEqual(
            row,
            ["1798876800000", "", "", "", "bytetrack", "0019", "car", "HOTA", "0.5"],
        )

    def test_rejects_non_finite_trackeval_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracker_folder = Path(directory) / "bytetrack"
            tracker_folder.mkdir()
            detailed_file = tracker_folder / "car_detailed.csv"
            detailed_file.write_text(
                "seq,"
                + ",".join(
                    [
                        "HOTA___AUC",
                        "DetA___AUC",
                        "AssA___AUC",
                        "DetRe___AUC",
                        "DetPr___AUC",
                        "AssRe___AUC",
                        "AssPr___AUC",
                        "LocA___AUC",
                        "OWTA___AUC",
                        "HOTA(0)",
                        "LocA(0)",
                        "HOTALocA(0)",
                        "MOTA",
                        "MOTP",
                        "MODA",
                        "CLR_Re",
                        "CLR_Pr",
                        "MTR",
                        "PTR",
                        "MLR",
                        "CLR_TP",
                        "CLR_FN",
                        "CLR_FP",
                        "IDSW",
                        "MT",
                        "PT",
                        "ML",
                        "Frag",
                        "sMOTA",
                        "IDF1",
                        "IDR",
                        "IDP",
                        "IDTP",
                        "IDFN",
                        "IDFP",
                        "Dets",
                        "GT_Dets",
                        "IDs",
                        "GT_IDs",
                    ]
                )
                + "\n0019,nan,"
                + ",".join(["0"] * 38)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite value"):
                read_trackeval_samples(Path(directory))


if __name__ == "__main__":
    unittest.main()
