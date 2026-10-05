import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
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
    model = YOLO(args.model_path, task="detect").to(device)

    # print(f"YOLO {model.names=}")

    # Resize KITTI images to nearest multiple of 32 for YOLO
    # FIXME: resizing inputs => BB coords will be different from
    # the full res BB coords in the KITTI label files
    new_hw = (
        math.ceil(args.scale * (KittiImages.HEIGHT // 32)) * 32,
        math.ceil(args.scale * (KittiImages.WIDTH // 32)) * 32,
    )

    scale_x = KittiImages.WIDTH / new_hw[1]
    scale_y = KittiImages.HEIGHT / new_hw[0]
    print(f"scale_xy = {(scale_x, scale_y)}")

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
        for frame, img in tqdm(enumerate(loader), total=len(dataset)):
            results = model.track(img, tracker="bytetrack.yaml", persist=True, verbose=False)

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

                    # Rescale back to original KITTI resolution
                    left = min(left * scale_x, KittiImages.WIDTH)
                    right = min(right * scale_x, KittiImages.WIDTH)
                    top = min(top * scale_y, KittiImages.HEIGHT)
                    bottom = min(bottom * scale_y, KittiImages.HEIGHT)

                    # 1    frame        Frame within the sequence where the object appearers
                    # 1    track id     Unique tracking id of this object within this sequence
                    # 1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                    #                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                    #                     'Misc' or 'DontCare'
                    # 1    truncated    Integer (0,1,2) indicating the level of truncation.
                    #                     Note that this is in contrast to the object detection
                    #                     benchmark where truncation is a float in [0,1].
                    # 1    occluded     Integer (0,1,2,3) indicating occlusion state:
                    #                     0 = fully visible, 1 = partly occluded
                    #                     2 = largely occluded, 3 = unknown
                    # 1    alpha        Observation angle of object, ranging [-pi..pi]
                    # 4    bbox         2D bounding box of object in the image (0-based index):
                    #                     contains left, top, right, bottom pixel coordinates
                    # 3    dimensions   3D object dimensions: height, width, length (in meters)
                    # 3    location     3D object location x,y,z in camera coordinates (in meters)
                    # 1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

                    # 1    score        Only for results: Float, indicating confidence in
                    #                     detection, needed for p/r curves, higher is better.

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
                        -1,
                        score,
                    ]
                    output.write(" ".join(map(str, row)) + "\n")

    print(f"Prediction written to: {outpath}")


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
        default="data/kitti/trackers/bytetrack/data",
        # required=False,
        type=str,
        help="Output folder for tracker predictions in KITTI TXT format.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Run the benchmark
    # TODO: Must I run this on all sequences?
    mot_metrics_to_kitti_file(args)
    # TODO: Use trackeval CLI w/ configs instead?
    # run_trackeval()
