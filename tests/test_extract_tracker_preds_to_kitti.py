import csv
import tempfile
import unittest
from pathlib import Path

from scripts.eval.extract_tracker_preds_to_kitti import (
    TrackBox,
    frame_from_timestamp,
    kitti_tracking_row,
    parse_track_label,
    tracks_from_components,
    write_kitti_tracks,
)


class RrdToKittiTests(unittest.TestCase):
    def test_frame_index_uses_frame_zero_timestamp_and_fps(self) -> None:
        self.assertEqual(frame_from_timestamp(1_200_000_000, 10.0, 1_000_000_000), 2)

    def test_parses_class_and_track_id_from_box_label(self) -> None:
        self.assertEqual(parse_track_label("Pedestrian #42"), ("Pedestrian", 42))
        with self.assertRaisesRegex(ValueError, "unrecognized track label"):
            parse_track_label("person 42")

    def test_converts_centers_and_half_sizes_to_kitti_box_coordinates(self) -> None:
        tracks = tracks_from_components(
            timestamp_ns=200_000_000,
            centers=[[100.0, 50.0]],
            half_sizes=[[20.0, 10.0]],
            labels=["Pedestrian #42"],
            fps=10.0,
            first_frame_timestamp_ns=0,
        )
        self.assertEqual(
            tracks,
            [TrackBox(frame=2, track_id=42, class_name="Pedestrian", left=80.0, top=40.0, right=120.0, bottom=60.0)],
        )

    def test_kitti_row_contains_standard_2d_tracking_fields_and_placeholders(self) -> None:
        row = kitti_tracking_row(
            TrackBox(frame=2, track_id=42, class_name="Pedestrian", left=80.0, top=40.0, right=120.0, bottom=60.0)
        )
        self.assertEqual(
            row,
            [2, 42, "Pedestrian", 0, 0, 0, "80.000000", "40.000000", "120.000000", "60.000000", -1, -1, -1,
             -1, -1, -1, -1, 1.0],
        )

    def test_writer_sorts_by_frame_then_id_and_rejects_duplicate_ids(self) -> None:
        tracks = [
            TrackBox(3, 10, "Car", 0.0, 0.0, 2.0, 2.0),
            TrackBox(1, 4, "Pedestrian", 0.0, 0.0, 2.0, 2.0),
            TrackBox(1, 2, "Car", 0.0, 0.0, 2.0, 2.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "0019.txt"
            write_kitti_tracks(tracks, output_path)
            with output_path.open(newline="", encoding="utf-8") as output_file:
                rows = list(csv.reader(output_file, delimiter=" "))
        self.assertEqual([(int(row[0]), int(row[1])) for row in rows], [(1, 2), (1, 4), (3, 10)])

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate track ID"):
                write_kitti_tracks([tracks[0], tracks[0]], Path(directory) / "duplicate.txt")


if __name__ == "__main__":
    unittest.main()
