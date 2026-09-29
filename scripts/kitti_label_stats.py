#!/usr/bin/env python3
"""
Analyze KITTI Tracking label files in a directory.

For each .txt label file computes:
  - sequence length (max frame index + 1)
  - number of occurrences of every object type

Usage:
    python analyze_kitti_labels.py /path/to/label_02
"""

from pathlib import Path
import argparse
import pandas as pd


def analyze_label_file(path: Path) -> dict:
    """Parse one KITTI tracking label file and return stats."""
    # Columns we care about (KITTI tracking format)
    # frame, track_id, type, truncated, occluded, alpha,
    # bbox_left, bbox_top, bbox_right, bbox_bottom,
    # height, width, length, x, y, z, rotation_y
    cols = [
        "frame",
        "track_id",
        "type",
        "truncated",
        "occluded",
        "alpha",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "height",
        "width",
        "length",
        "x",
        "y",
        "z",
        "rotation_y",
    ]

    try:
        # "\s+" 1 or more consecutive whitespace
        df = pd.read_csv(path, sep=r"\s+", header=None, names=cols, engine="python")
    except Exception as e:
        print(f"Warning: could not parse {path.name}: {e}")
        return None

    if df.empty:
        return {
            "sequence": path.stem,
            "length": 0,
        }

    # Sequence length = highest frame index + 1 (0-based)
    length = int(df["frame"].max()) + 1

    # Count occurrences of each object type
    type_counts = df["type"].value_counts().to_dict()

    result = {"sequence": path.stem, "length": length, **type_counts}
    return result


def main(label_dir: str):
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise SystemExit(f"Directory not found: {label_dir}")

    files = sorted(label_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {label_dir}")

    rows = []
    for f in files:
        stats = analyze_label_file(f)
        if stats is not None:
            rows.append(stats)

    # Build DataFrame; missing type columns become NaN → fill with 0
    df = pd.DataFrame(rows).fillna(0)

    # Ensure integer counts
    count_cols = [c for c in df.columns if c not in ("sequence", "length")]
    df[count_cols] = df[count_cols].astype(int)

    # Nice column order: sequence, length, then alphabetical types
    ordered = ["sequence", "length"] + sorted(count_cols)
    df = df[ordered]

    # Optional: total objects column
    df["total_objects"] = df[count_cols].sum(axis=1)

    print(df.to_string(index=False))
    print(f"\nProcessed {len(df)} sequences from {label_dir}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI label sequence statistics")
    parser.add_argument(
        "label_dir",
        type=str,
        help="Directory containing KITTI tracking label .txt files (e.g. label_02)",
    )
    args = parser.parse_args()
    main(args.label_dir)
