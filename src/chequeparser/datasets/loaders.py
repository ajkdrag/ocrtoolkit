from pathlib import Path
from typing import List, Tuple

import yaml
from loguru import logger

from chequeparser.datasets.fileds import FileDS
from chequeparser.utilities.io_utils import change_suffixes, get_files
from chequeparser.wrappers.bbox import BBox
from chequeparser.wrappers.detection_results import DetectionResults


def load_yolo(path: str, subset="train") -> Tuple["BaseDS", List["DetectionResults"]]:
    """Takes input the dataset.yml file path, which contains
    detections in YOLO format and returns a tuple of
    (BaseDS, List[DetectionResults])
    """
    l_dets = []
    with open(path, "r") as f:
        d = yaml.safe_load(f)
        class_names = d["names"]
        p_images = Path(d[subset])
        p_labels = p_images.parent.joinpath("labels")
        logger.info(f"Loading labels from {p_labels}")
        all_labels = get_files(p_labels, exts=[".txt"])
        corr_images = change_suffixes(all_labels, ".jpg", ref_dir=p_images)
        ds = FileDS(items=corr_images, size=None)

        for f_img, f_label in zip(corr_images, all_labels):
            img_name = f_img.name
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

        return l_dets, ds
