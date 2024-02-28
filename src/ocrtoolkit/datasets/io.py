from pathlib import Path
from typing import List, Tuple

import yaml
from loguru import logger

from ocrtoolkit.datasets.base import BaseDS
from ocrtoolkit.datasets.fileds import FileDS
from ocrtoolkit.utilities.io_utils import get_files
from ocrtoolkit.wrappers.bbox import BBox
from ocrtoolkit.wrappers.detection_results import DetectionResults


def _read_image_paths_from_txt_or_dir(paths: List[str], root_path: str):
    image_paths = []
    p_root = Path(root_path)
    for path in paths:
        if path.endswith(".txt"):
            with p_root.joinpath(path).open() as f:
                image_paths.extend([line.strip() for line in f.readlines()])
        else:
            p_images = p_root.joinpath(path)
            image_paths.extend(get_files(p_images))
    return image_paths


def _get_label_path(image_path: str, subdir="labels"):
    parent_dir = Path(image_path).parent
    label_dir = parent_dir.parent.joinpath(subdir)
    label_file = Path(image_path).with_suffix(".txt").name
    return label_dir.joinpath(label_file)


def load_yolo(path: str, subset="train") -> Tuple["BaseDS", List["DetectionResults"]]:
    """Takes input the dataset.yml file path, which contains
    detections in YOLO format (xywh) and returns a tuple of
    (BaseDS, List[DetectionResults])
    """
    with open(path, "r") as f:
        dataset_info = yaml.safe_load(f)

    root_path = Path(dataset_info["path"])
    assert root_path.exists(), f"{root_path} does not exist"

    images = _read_image_paths_from_txt_or_dir(
        dataset_info.get(subset, []), root_path.as_posix()
    )
    labels = [_get_label_path(img) for img in images]
    filtered_images_and_labels = [
        (img, label) for img, label in zip(images, labels) if Path(label).exists()
    ]
    valid_images, valid_labels = zip(*filtered_images_and_labels)
    logger.info(f"Found {len(valid_images)} valid images out of {len(images)}")

    ds = FileDS(items=valid_images, size=None)
    l_dets = []
    class_names = dataset_info["names"]

    for f_img, f_label in filtered_images_and_labels:
        img_name = Path(f_img).name
        img = ds[img_name]
        l_str_bboxes = open(f_label, "r").readlines()
        l_bboxes = []
        for str_bbox in l_str_bboxes:
            l_bbox = [float(x) for x in str_bbox.split(" ")]
            class_ = class_names[int(l_bbox[0])]
            l_bbox = BBox.from_cxcywh(
                *l_bbox[1:],
                normalized=True,
                label=class_,
            ).denormalize(img.width, img.height)
            l_bboxes.append(l_bbox)
        l_dets.append(DetectionResults(l_bboxes, img.width, img.height, img_name))

    return ds, l_dets
