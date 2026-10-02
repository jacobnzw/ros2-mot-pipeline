import argparse
import math
from pathlib import Path

import torch
import trackeval
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

YOLOv11n = "models/yolo/yolo11n.pt"


class KittiImages(Dataset):
    WIDTH = 1238
    HEIGHT = 374

    def __init__(self, img_dir: str, ext: str = "png", transform=None):
        self.paths = sorted(Path(img_dir).glob(f"*.{ext}"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):  # ty: ignore[invalid-method-override]
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def mot_metrics_to_kitti_file(args):

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = YOLO(args.model_path).to(device)

    print(f"YOLO {model.names=}")

    # Resize KITTI images to nearest multiple of 32 for YOLO
    # FIXME: resizing inputs => BB coords will be different from
    # the full res BB coords in the KITTI label files
    new_hw = (
        math.ceil(args.scale * (KittiImages.HEIGHT // 32)) * 32,
        math.ceil(args.scale * (KittiImages.WIDTH // 32)) * 32,
    )
    transform = transforms.Compose(
        [
            transforms.Resize(new_hw),
            transforms.ToTensor(),
        ]
    )

    dataset = KittiImages(args.kitti_seq, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # TODO: check this!
    # class_ids = {"car": 0, "person": 3}
    class_ids: dict[str, str] = {
        "car": "Car",
        # "van": "Van",
        # "truck": "Truck",
        "person": "Pedestrian",
    }

    seq_id = Path(args.kitti_seq).name
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{seq_id}.txt"

    with open(outpath, "w") as output:
        for frame, img in enumerate(loader):
            results = model.track(img, tracker="bytetrack.yaml", verbose=False)

            for result in results:
                boxes = result.boxes
                if boxes.id is None:
                    continue

                track_ids = boxes.id.int().cpu().tolist()
                model_class_ids = boxes.cls.int().cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for track_id, model_class_id, (left, top, right, bottom), score in zip(
                    track_ids, model_class_ids, xyxys, confidences
                ):
                    class_name = str(model.names[model_class_id]).lower()
                    kitti_class_id = class_ids.get(class_name, "DontCare")
                    if kitti_class_id is None:
                        continue

                    row = [
                        frame,
                        track_id,
                        kitti_class_id,
                        0,
                        0,
                        0,
                        left,
                        top,
                        right,
                        bottom,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        score,
                    ]
                    output.write(" ".join(map(str, row)) + "\n")

    print(f"Prediction written to: {outpath}")


def run_trackeval():
    """
    What TrackEval expects for KITTI

    Dataset class: trackeval/datasets/kitti_2d_box.py
    GT folder by default: data/gt/kitti/kitti_2d_box_train
    Tracker folder by default: data/trackers/kitti/kitti_2d_box_train
    Sequence list: evaluate_tracking.seqmap.training / val / etc.
    Per tracker: one file per sequence under: data/trackers/kitti/kitti_2d_box_train/<tracker_name>/data/0000.txt, 0001.txt, ...
    For each sequence, each line is a detection/track result in KITTI-style format.
    The tracker output format is similar to KITTI labels, but in TrackEval it mainly reads:

    frame index
    track ID
    class ID
    bbox: left, top, right, bottom
    confidence at the end (optional but recommended)
    The dataset code explicitly reads:

    time column = 0
    id column = 1
    class column = 2
    bbox columns = 6:10
    optional confidence = column 17

    So your ByteTrack output needs to be converted to something like:

    frame, track_id, class_id, truncation, occlusion, alpha, left, top, right, bottom,
    dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, score

    For example:

    car class id = 1
    pedestrian class id = 4
    truncation/occlusion usually 0
    the bbox is x1, y1, x2, y2
    confidence in the final column

    A typical KITTI-style row would be roughly:
    0, 7, 1, 0, 0, 0, 292.0, 159.5, 364.0, 208.0, -1, -1, -1, -1, -1, -1, 0.91

    Important practical note

    For KITTI, TrackEval has class-specific preprocessing. It validates against:
    car
    pedestrian
    and some distractor mappings like van/person
    If your ByteTrack output contains only one class, you can just evaluate that class.
    If it outputs both car and pedestrian, put both in CLASSES_TO_EVAL and ensure IDs match the KITTI mapping:
    car = 1
    pedestrian = 4
    """

    code_path: Path = Path(trackeval.utils.get_code_path())

    eval_config = {
        "USE_PARALLEL": False,
        "PRINT_RESULTS": True,
        "OUTPUT_SUMMARY": True,
    }

    dataset_config = {
        "GT_FOLDER": code_path / "data/kitti/left_color/training/label_02",
        # Tracker predictions in a text file in KITTI format
        "TRACKERS_FOLDER": code_path / "data/kitti/trackers",
        "TRACKERS_TO_EVAL": ["ByteTrack"],
        "SPLIT_TO_EVAL": "training",
        "CLASSES_TO_EVAL": ["car", "person"],
    }

    dataset = trackeval.datasets.Kitti2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator = trackeval.Evaluator(eval_config)
    evaluator.evaluate([dataset], metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference-call latency of YOLO + ByteTrack.")
    parser.add_argument(
        "--model-path",
        default=YOLOv11n,
        # required=False,
        help="Path to YOLO model *.pt file",
    )
    parser.add_argument(
        "--kitti-seq",
        type=str,
        default="",
        required=False,
        help="Path to KITTI sequence.",
    )
    parser.add_argument(
        "--outdir",
        default="data/kitti/trackers",
        # required=False,
        type=str,
        help="Output folder for tracker predictions in KITTI TXT format.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Run the benchmark
    mot_metrics_to_kitti_file(args)
