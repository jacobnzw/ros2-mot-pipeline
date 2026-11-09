#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from tqdm import tqdm

NANOSECONDS = 1_000_000_000
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def find_images(image_dir: Path) -> list[Path]:
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return images


def write_sequence(
    images: list[Path],
    output: Path,
    fps: float,
    topic: str,
    frame_id: str,
    typestore: Typestore,
) -> None:
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]
    msgtype = CompressedImage.__msgtype__

    with Writer(
        output,
        version=9,
        storage_plugin=StoragePlugin.MCAP,
    ) as writer:
        connection = writer.add_connection(topic, msgtype, typestore=typestore)

        image_progress = tqdm(
            enumerate(images),
            total=len(images),
            desc=output.name,
            unit="image",
            leave=False,
        )
        for index, image_path in image_progress:
            timestamp_ns = round(index * NANOSECONDS / fps)
            seconds, nanoseconds = divmod(timestamp_ns, NANOSECONDS)
            image_format = "png" if image_path.suffix.lower() == ".png" else "jpeg"

            message = CompressedImage(
                header=Header(
                    stamp=Time(sec=seconds, nanosec=nanoseconds),
                    frame_id=frame_id,
                ),
                format=image_format,
                data=np.fromfile(image_path, dtype=np.uint8),
            )
            writer.write(
                connection,
                timestamp_ns,
                typestore.serialize_cdr(message, msgtype),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write each image sequence in a directory to its own ROS 2 MCAP bag.")
    parser.add_argument(
        "sequence_root",
        type=Path,
        help="Folder containing one subfolder per sequence, each with PNG or JPEG images",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output folder for sequence rosbag directories (must not exist)",
    )
    parser.add_argument("--topic", default="/camera/image_raw/compressed")
    parser.add_argument("--frame-id", default="camera")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    # Expand ~ to user home dir
    sequence_root = args.sequence_root.expanduser()
    output_root = args.output_root.expanduser()

    if not sequence_root.is_dir():
        parser.error(f"Sequence folder does not exist: {sequence_root}")
    if output_root.exists():
        parser.error(f"Output path already exists: {output_root}")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and greater than zero")

    sequence_dirs = sorted(path for path in sequence_root.iterdir() if path.is_dir())
    if not sequence_dirs:
        parser.error(f"No sequence folders found in {sequence_root}")

    sequences = [(sequence_dir, find_images(sequence_dir)) for sequence_dir in sequence_dirs]
    empty_sequences = [sequence_dir for sequence_dir, images in sequences if not images]
    if empty_sequences:
        parser.error(
            "No PNG or JPEG images found in sequence folder(s): "
            + ", ".join(str(sequence_dir) for sequence_dir in empty_sequences)
        )

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    output_root.mkdir(parents=True)
    for sequence_dir, images in tqdm(sequences, desc="Sequences", unit="sequence"):
        output = output_root / sequence_dir.name
        write_sequence(images, output, args.fps, args.topic, args.frame_id, typestore)
        tqdm.write(f"Wrote {len(images)} images to {output}")


if __name__ == "__main__":
    main()
