#!/usr/bin/env python3

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import rerun_bindings

TRACK_ENTITY_PATH = "/tracker/tracks"
ROS2_TIMESTAMP_COLUMN = "ros2_timestamp"
CENTERS_COLUMN = "Boxes2D:centers"
HALF_SIZES_COLUMN = "Boxes2D:half_sizes"
LABELS_COLUMN = "Boxes2D:labels"
NANOSECONDS = 1_000_000_000
TRACK_LABEL_PATTERN = re.compile(r"^(Pedestrian|Car|Truck|Cyclist|Tram) #([0-9]+)$")


@dataclass(frozen=True)
class TrackBox:
    frame: int
    track_id: int
    class_name: str
    left: float
    top: float
    right: float
    bottom: float


def frame_from_timestamp(timestamp_ns: int, fps: float, first_frame_timestamp_ns: int) -> int:
    if timestamp_ns < first_frame_timestamp_ns:
        raise ValueError(
            f"track timestamp {timestamp_ns} is earlier than frame-zero timestamp {first_frame_timestamp_ns}"
        )
    return round((timestamp_ns - first_frame_timestamp_ns) * fps / NANOSECONDS)


def parse_track_label(label: str) -> tuple[str, int]:
    match = TRACK_LABEL_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"unrecognized track label {label!r}; expected 'ClassName #track_id'")
    return match.group(1), int(match.group(2))


def tracks_from_components(
    timestamp_ns: int,
    centers: list[list[float]],
    half_sizes: list[list[float]],
    labels: list[str],
    fps: float,
    first_frame_timestamp_ns: int,
) -> list[TrackBox]:
    if len(centers) != len(half_sizes) or len(centers) != len(labels):
        raise ValueError("track centers, half-sizes, and labels must have matching lengths")

    frame = frame_from_timestamp(timestamp_ns, fps, first_frame_timestamp_ns)
    tracks = []
    for center, half_size, label in zip(centers, half_sizes, labels, strict=True):
        if len(center) != 2 or len(half_size) != 2:
            raise ValueError("each track center and half-size must contain exactly two coordinates")
        center_x, center_y = center
        half_width, half_height = half_size
        if not all(math.isfinite(value) for value in (center_x, center_y, half_width, half_height)):
            raise ValueError(f"track {label!r} has non-finite box coordinates")
        if half_width <= 0 or half_height <= 0:
            raise ValueError(f"track {label!r} has non-positive box dimensions")

        class_name, track_id = parse_track_label(label)
        tracks.append(
            TrackBox(
                frame=frame,
                track_id=track_id,
                class_name=class_name,
                left=center_x - half_width,
                top=center_y - half_height,
                right=center_x + half_width,
                bottom=center_y + half_height,
            )
        )
    return tracks


def read_tracks(recording_path: Path, fps: float, first_frame_timestamp_ns: int) -> list[TrackBox]:
    reader = rerun_bindings.RrdReaderInternal(str(recording_path))
    stream = rr.chunk.LazyChunkStream(reader.stream())
    tracks = []
    found_track_entity = False

    for chunk in stream:
        if chunk.entity_path != TRACK_ENTITY_PATH:
            continue
        found_track_entity = True
        batch = chunk.to_record_batch()
        columns = set(batch.schema.names)
        if CENTERS_COLUMN not in columns:
            continue

        required_columns = {ROS2_TIMESTAMP_COLUMN, HALF_SIZES_COLUMN, LABELS_COLUMN}
        missing_columns = required_columns - columns
        if missing_columns:
            raise ValueError(f"{TRACK_ENTITY_PATH} is missing RRD components: {', '.join(sorted(missing_columns))}")

        timestamps = batch.column(ROS2_TIMESTAMP_COLUMN)
        centers_column = batch.column(CENTERS_COLUMN)
        half_sizes_column = batch.column(HALF_SIZES_COLUMN)
        labels_column = batch.column(LABELS_COLUMN)
        for row_index in range(batch.num_rows):
            timestamp = timestamps[row_index]
            centers = centers_column[row_index].as_py()
            half_sizes = half_sizes_column[row_index].as_py()
            labels = labels_column[row_index].as_py()
            if not timestamp.is_valid or centers is None or half_sizes is None or labels is None:
                continue
            tracks.extend(
                tracks_from_components(
                    timestamp.value,
                    centers,
                    half_sizes,
                    labels,
                    fps,
                    first_frame_timestamp_ns,
                )
            )

    if not found_track_entity:
        raise ValueError(f"recording has no {TRACK_ENTITY_PATH} entity")
    return tracks


def kitti_tracking_row(track: TrackBox) -> list[str | int | float]:
    # KITTI tracking columns: frame, ID, type, truncation, occlusion, alpha,
    # 2D box, 3D dimensions, 3D location, rotation_y, and optional score.
    return [
        track.frame,
        track.track_id,
        track.class_name,
        0,
        0,
        0,
        *(f"{value:.6f}" for value in (track.left, track.top, track.right, track.bottom)),
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1.0,
    ]


def write_kitti_tracks(tracks: list[TrackBox], output_path: Path) -> None:
    ordered_tracks = sorted(tracks, key=lambda track: (track.frame, track.track_id))
    seen_track_ids: set[tuple[int, int]] = set()
    for track in ordered_tracks:
        key = (track.frame, track.track_id)
        if key in seen_track_ids:
            raise ValueError(f"duplicate track ID {track.track_id} in frame {track.frame}")
        seen_track_ids.add(key)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, delimiter=" ", lineterminator="\n")
        for track in ordered_tracks:
            writer.writerow(kitti_tracking_row(track))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export tracker/tracks boxes and labels from a Rerun recording as KITTI tracking rows."
    )
    parser.add_argument("recording", type=Path, help="Input Rerun .rrd file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output KITTI tracking text file (default: <recording-stem>.txt beside the recording)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Source sequence frame rate used to map ROS timestamps to frame indices (default: 10)",
    )
    parser.add_argument(
        "--first-frame-timestamp-ns",
        type=int,
        default=0,
        help="ROS timestamp of sequence frame 0 in nanoseconds (default: 0, as written by kitti_to_mcap.py)",
    )
    args = parser.parse_args()

    recording_path = args.recording.expanduser()
    if not recording_path.is_file():
        parser.error(f"recording file does not exist: {recording_path}")
    if recording_path.suffix.lower() != ".rrd":
        parser.error(f"expected a .rrd recording: {recording_path}")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and greater than zero")

    output_path = args.output.expanduser() if args.output else recording_path.with_suffix(".txt")
    if output_path.resolve() == recording_path.resolve():
        parser.error("output text file must not overwrite the input recording")

    try:
        tracks = read_tracks(recording_path, args.fps, args.first_frame_timestamp_ns)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_kitti_tracks(tracks, output_path)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Exported {len(tracks):,} tracks across {len({track.frame for track in tracks}):,} frames to {output_path}")


if __name__ == "__main__":
    main()
