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
        all_images = get_files(p_images)
        corr_labels = change_suffixes(all_images, ".txt", ref_dir=p_labels)
        all_labels_wo_ext = set([Path(label).stem for label in corr_labels])
        valid_images = [img for img in all_images if Path(img).stem in all_labels_wo_ext]
        ds = FileDS(items=valid_images, size=None)

        for f_img, f_label in zip(valid_images, corr_labels):
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
