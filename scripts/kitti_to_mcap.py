#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

NANOSECONDS = 1_000_000_000
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an image folder to a ROS 2 MCAP bag."
    )
    parser.add_argument(
        "image_dir", type=Path, help="Folder containing PNG or JPEG images"
    )
    parser.add_argument(
        "output", type=Path, help="Output rosbag directory (must not exist)"
    )
    parser.add_argument("--topic", default="/camera/image_raw/compressed")
    parser.add_argument("--frame-id", default="camera")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    image_dir = args.image_dir.expanduser()
    output = args.output.expanduser()

    # Check args
    if not image_dir.is_dir():
        parser.error(f"Image folder does not exist: {image_dir}")
    if output.exists():
        parser.error(f"Output path already exists: {output}")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    # Grab images to process
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        parser.error(f"No PNG or JPEG images found in {image_dir}")

    # Write to MCAP
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]
    msgtype = CompressedImage.__msgtype__

    with Writer(
        output,
        version=9,
        storage_plugin=StoragePlugin.MCAP,
    ) as writer:
        connection = writer.add_connection(
            args.topic,
            msgtype,
            typestore=typestore,
        )

        for index, image_path in enumerate(images):
            timestamp_ns = round(index * NANOSECONDS / args.fps)
            seconds, nanoseconds = divmod(timestamp_ns, NANOSECONDS)
            image_format = "png" if image_path.suffix.lower() == ".png" else "jpeg"

            message = CompressedImage(
                header=Header(
                    stamp=Time(sec=seconds, nanosec=nanoseconds),
                    frame_id=args.frame_id,
                ),
                format=image_format,
                data=np.fromfile(image_path, dtype=np.uint8),
            )
            writer.write(
                connection,
                timestamp_ns,
                typestore.serialize_cdr(message, msgtype),
            )

    print(f"Wrote {len(images)} images to {output}")


if __name__ == "__main__":
    main()
