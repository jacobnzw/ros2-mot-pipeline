#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import yaml
from rosbags.rosbag2 import Reader

DEFAULT_KITTI_ROOT = Path("data/kitti/left_color/training/image_02_mcap")
DEFAULT_KITTI_GT = Path("data/kitti/left_color/training/label_02")
TRACKER_GT_SPLIT = "training"
IMAGE_MESSAGE_TYPE = "sensor_msgs/msg/CompressedImage"


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_sequences(value: str) -> list[str]:
    sequences = [sequence.strip() for sequence in value.split(",")]
    if not sequences or any(not sequence for sequence in sequences):
        raise argparse.ArgumentTypeError("must be a comma-separated list of non-empty sequence numbers")
    if len(sequences) != len(set(sequences)):
        raise argparse.ArgumentTypeError("sequence numbers must not be repeated")
    return sequences


# TODO: generalize suitably (don't assume rtx3090)
def identify_hardware(device_tree_model: str | None, gpu_name: str | None) -> str:
    if device_tree_model and "jetson" in device_tree_model.lower():
        return "jetson"
    if gpu_name and "rtx 3090" in gpu_name.lower():
        return "rtx3090"
    raise ValueError("unsupported or unidentified GPU; expected a Jetson or RTX 3090")


def detect_hardware() -> str:
    device_tree_model: str | None = None
    try:
        device_tree_model = Path("/proc/device-tree/model").read_text(encoding="utf-8").replace("\0", "")
    except OSError:
        pass

    if device_tree_model and "jetson" in device_tree_model.lower():
        return "jetson"
    if Path("/etc/nv_tegra_release").is_file():
        return "jetson"

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu_name = result.stdout if result.returncode == 0 else None

    return identify_hardware(device_tree_model, gpu_name)


def discover_sequences(kitti_root: Path, requested_sequences: list[str] | None = None) -> list[Path]:
    if not kitti_root.is_dir():
        raise ValueError(f"KITTI sequence root does not exist: {kitti_root}")

    available_sequences = sorted(path for path in kitti_root.iterdir() if path.is_dir())
    if not available_sequences:
        raise ValueError(f"No sequence directories found in {kitti_root}")

    if requested_sequences is None:
        sequences = available_sequences
    else:
        available_by_name = {path.name: path for path in available_sequences}
        unknown_sequences = [sequence for sequence in requested_sequences if sequence not in available_by_name]
        if unknown_sequences:
            raise ValueError(f"Unknown sequence(s) in {kitti_root}: {', '.join(unknown_sequences)}")
        sequences = [available_by_name[sequence] for sequence in requested_sequences]

    missing_mcaps = [
        sequence / f"{sequence.name}.mcap"
        for sequence in sequences
        if not (sequence / f"{sequence.name}.mcap").is_file()
    ]
    if missing_mcaps:
        paths = ", ".join(str(path) for path in missing_mcaps)
        raise ValueError(f"Missing MCAP file(s): {paths}")
    return sequences


def count_mcap_frames(mcap_path: Path) -> int:
    with Reader(mcap_path) as reader:
        image_connections = [
            connection for connection in reader.connections if connection.msgtype == IMAGE_MESSAGE_TYPE
        ]
        if len(image_connections) != 1:
            raise ValueError(
                f"expected exactly one {IMAGE_MESSAGE_TYPE} connection in {mcap_path}, found {len(image_connections)}"
            )
        frame_count = sum(1 for _ in reader.messages(connections=image_connections))
        # TODO: use reader.message_count ??

    if frame_count == 0:
        raise ValueError(f"MCAP contains no image frames: {mcap_path}")
    return frame_count


def create_trackeval_gt(
    eval_folder: Path,
    kitti_gt: Path,
    sequences: list[Path],
) -> Path:
    label_dir = kitti_gt / "label_02" if (kitti_gt / "label_02").is_dir() else kitti_gt
    if not label_dir.is_dir():
        raise ValueError(f"KITTI ground-truth label folder does not exist: {label_dir}")

    seqmap_entries = []
    for sequence_dir in sequences:
        sequence = sequence_dir.name
        frame_count = count_mcap_frames(sequence_dir / f"{sequence}.mcap")
        seqmap_entries.append(f"{sequence} image_02/{sequence} 000000 {frame_count}")

    eval_folder.mkdir(parents=True, exist_ok=True)
    label_link = eval_folder / "label_02"
    if label_link.is_symlink():
        if label_link.resolve() != label_dir.resolve():
            raise ValueError(f"existing label_02 symlink points to a different directory: {label_link}")
    elif label_link.exists():
        raise ValueError(f"expected label_02 to be a symlink, but it already exists: {label_link}")
    else:
        label_link.symlink_to(label_dir.resolve(), target_is_directory=True)

    seqmap_path = eval_folder / f"evaluate_tracking.seqmap.{TRACKER_GT_SPLIT}"
    seqmap_path.write_text("\n".join(seqmap_entries) + "\n", encoding="utf-8")
    return seqmap_path


def write_metadata(
    metadata_path: Path,
    kitti_root: Path,
    kitti_gt: Path,
    trackeval_gt: Path,
    trackeval_seqmap: Path,
    thread_num: int,
    hardware: str,
    precision: str,
    label: str,
) -> None:
    metadata = {
        "thread_num": thread_num,
        "hardware": hardware,
        "precision": precision,
        "label": label,
        "label_dir": f"runs/{label}",
        "kitti_gt": f"{kitti_gt}",
        "kitti_root": f"{kitti_root}",
        "trackeval_gt": str(trackeval_gt.resolve()),  # TODO: can't we just save to kitti_gt instead? extra field?
        "trackeval_seqmap": str(trackeval_seqmap.resolve()),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        yaml.safe_dump(metadata, metadata_file, sort_keys=False)


def launch_command(
    sequence: str,
    kitti_root: Path,
    thread_num: int,
    precision: str,
    label: str,
) -> list[str]:
    return [
        "ros2",
        "launch",
        "launch/detector_test.launch.py",
        f"thread_num:={thread_num}",
        f"seq:={sequence}",
        f"label:={label}",
        f"precision:={precision}",
        f"kitti_root:={kitti_root}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the detector launch sequentially for each KITTI sequence MCAP.")
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT, help="Root containing sequence folders")
    parser.add_argument(
        "--kitti-gt", type=Path, default=DEFAULT_KITTI_GT, help="Root containing ground truth TXT file per sequence."
    )
    parser.add_argument("--thread-num", type=positive_int, required=True, help="Detector container thread count")
    parser.add_argument("--precision", required=True, help="Detector model export precision, e.g. fp32 or fp16")
    parser.add_argument("--label", required=True, help="Run label used for recording and metadata directories")
    parser.add_argument(
        "--seq",
        type=parse_sequences,
        help="Comma-separated sequence numbers to run (default: all sequence folders)",
    )
    args = parser.parse_args()

    if not args.label or args.label in {".", ".."} or "/" in args.label or "\\" in args.label:
        parser.error("--label must be a non-empty directory name without path separators")

    kitti_root = args.kitti_root.expanduser().resolve()
    kitti_gt = args.kitti_gt.expanduser().resolve()
    try:
        sequences = discover_sequences(kitti_root, args.seq)
        hardware = detect_hardware()
    except (OSError, ValueError) as error:
        parser.error(str(error))

    repository_root = Path(__file__).resolve().parents[2]
    try:
        run_dir = repository_root / "runs" / args.label
        eval_folder = run_dir / "eval"
        seqmap_path = create_trackeval_gt(eval_folder, kitti_gt, sequences)
        metadata_path = run_dir / "metadata.yaml"
        write_metadata(
            metadata_path,
            kitti_root,
            kitti_gt,
            eval_folder,
            seqmap_path,
            args.thread_num,
            hardware,
            args.precision,
            args.label,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(f"could not prepare run metadata or TrackEval ground truth: {error}")

    print(f"Hardware: {hardware}")
    print(f"Metadata: {metadata_path}")
    for sequence in sequences:
        command = launch_command(sequence.name, kitti_root, args.thread_num, args.precision, args.label)
        print(f"\n==> Running KITTI sequence {sequence.name}", flush=True)
        result = subprocess.run(command, cwd=repository_root, check=False)
        if result.returncode != 0:
            print(f"Error: launch failed for sequence {sequence.name} (exit code {result.returncode})")
            return result.returncode

    print(f"Completed {len(sequences)} sequence(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
