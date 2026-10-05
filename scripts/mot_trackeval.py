import trackeval
from pathlib import Path


def run_trackeval():

    # code_path: Path = Path(trackeval.utils.get_code_path())

    eval_config = {
        "USE_PARALLEL": False,
        "PRINT_RESULTS": True,
        "OUTPUT_SUMMARY": True,
    }

    dataset_config = {
        "GT_FOLDER": "data/kitti/left_color/training",
        # Tracker predictions in a text file in KITTI format
        "TRACKERS_FOLDER": "data/kitti/trackers",
        "TRACKERS_TO_EVAL": ["bytetrack"],
        "SPLIT_TO_EVAL": "training",
        "CLASSES_TO_EVAL": ["car", "pedestrian"],  # TODO: presumably the KITTI object types??
    }

    dataset = trackeval.datasets.Kitti2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator = trackeval.Evaluator(eval_config)
    evaluator.evaluate([dataset], metrics)


if __name__ == "__main__":
    run_trackeval()
