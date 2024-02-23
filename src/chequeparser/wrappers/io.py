import json
from pathlib import Path

import h5py
from loguru import logger

from chequeparser.wrappers.bbox import BBox
from chequeparser.wrappers.detection_results import DetectionResults


def save_dets(l_dets, path: str):
    with h5py.File(path, "w") as f:
        group = f.create_group("dets")
        for idx, dets in enumerate(l_dets):
            npy_bboxes = dets.to_numpy()
            dset = group.create_dataset(f"dets_{idx}", data=npy_bboxes)
            dset.attrs["width"] = dets.width
            dset.attrs["height"] = dets.height
            dset.attrs["img_name"] = dets.img_name
        logger.info(f"Detections saved to {path}")


def save_dets_as_label_studio(l_dets, path: str, subdir_images="images"):
    """Save detections as Label Studio json format"""
    base_dir = "/data/local-files/?d={subdir_images}"
    l_json_data = [
        {
            "data": {
                "image": base_dir.format(
                    subdir_images=Path(subdir_images)
                    .joinpath(detection.img_name)
                    .as_posix()
                ),
            },
            "predictions": [
                {
                    "model_version": "one",
                    "score": 0.5,
                    "result": [
                        {
                            "id": f"bbox{i+1}",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": detection.width,
                            "original_height": detection.height,
                            "image_rotation": 0,
                            "value": {
                                "rotation": 0,
                                "x": bbox.x1 * 100,
                                "y": bbox.y1 * 100,
                                "width": bbox.w * 100,
                                "height": bbox.h * 100,
                                "rectanglelabels": [bbox.label],
                            },
                        }
                        for i, bbox in enumerate(detection.normalize().bboxes)
                    ],
                }
            ],
        }
        for detection in l_dets
    ]
    with open(path, "w") as f:
        json.dump(l_json_data, f, indent=2)


def load_dets(path: str):
    with h5py.File(path, "r") as f:
        l_dets = []
        group = f["dets"]
        dets_keys = sorted(group.keys(), key=lambda x: int(x.split("_")[-1]))
        for key in dets_keys:
            dets_width = int(group[key].attrs["width"])
            dets_height = int(group[key].attrs["height"])
            dets_img_name = str(group[key].attrs["img_name"])
            dets_data = group[key][()]
            l_bboxes = [BBox.from_numpy(bbox) for bbox in dets_data]
            l_dets.append(
                DetectionResults(l_bboxes, dets_width, dets_height, dets_img_name)
            )
        return l_dets
