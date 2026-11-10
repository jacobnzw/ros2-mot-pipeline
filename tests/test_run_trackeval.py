import tempfile
import unittest
from pathlib import Path

from scripts.run_trackeval import read_sequence_entry


class MotTrackEvalTests(unittest.TestCase):
    def test_selects_sequence_entry_from_whitespace_seqmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seqmap = Path(directory) / "evaluate_tracking.seqmap.training"
            seqmap.write_text(
                "0018 /data/image_02/0018 000000 000837\n0019 /data/image_02/0019 000000 001059\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_sequence_entry(seqmap, "0019"),
                "0019 /data/image_02/0019 000000 001059",
            )

    def test_missing_sequence_reports_seqmap_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seqmap = Path(directory) / "evaluate_tracking.seqmap.training"
            seqmap.write_text("0019 /data/image_02/0019 000000 001059\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sequence '0018' is not listed"):
                read_sequence_entry(seqmap, "0018")

    def test_duplicate_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seqmap = Path(directory) / "evaluate_tracking.seqmap.training"
            seqmap.write_text(
                "0019 /data/image_02/0019 000000 001059\n0019 /data/image_02/other 000000 001059\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "appears more than once"):
                read_sequence_entry(seqmap, "0019")


if __name__ == "__main__":
    unittest.main()
