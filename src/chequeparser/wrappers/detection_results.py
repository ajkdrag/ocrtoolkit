from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from doctr.models.builder import DocumentBuilder

from chequeparser.datasets.base import BaseDS
from chequeparser.datasets.imageds import ImageDS
from chequeparser.utilities.draw_utils import draw_bbox
from chequeparser.utilities.misc import get_samples, get_uuid
from chequeparser.wrappers.bbox import BBox


class DetectionResults:
    """Wrapper for detection results from a single image
    Captures the resulting bbox detections into a BBox object
    Assumes all bboxes are denormalized
    Stores the original input image width and height
    Methods to draw the bbox on a canvas
    """

    bboxes: List[BBox]
    width: int
    height: int
    img_name: str

    def __init__(self, bboxes, width, height, img_name="", denormalize=True):
        self.bboxes = bboxes
        self.width = width
        self.height = height
        self.img_name = img_name
        if denormalize:
            self.bboxes = [bbox.denormalize(self.width, self.height) for bbox in bboxes]

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        return self.bboxes[idx]

    def normalize(self):
        """Results are denormalized"""
        d_bboxes = [bbox.normalize(self.width, self.height) for bbox in self.bboxes]
        return DetectionResults(
            d_bboxes, self.width, self.height, self.img_name, denormalize=False
        )

    def to_numpy(self, normalize=False, encode=False, include_meta=False) -> np.ndarray:
        """Returns bboxes as a numpy array
        Each bbox object is converted to a numpy array
        If normalize is True, the bboxes are normalized
        If N=number of bboxes, we have the following np array created:
        If include_meta is True, add img_name, width, height as well
        1. Nx7 array with the x1, y1, x2, y2, conf, normalized, label
        or
        2. Nx9 array with the x1, y1, x2, y2, conf, normalized, label,
        text, text_conf
        """
        bboxes = self.bboxes
        if normalize:
            bboxes = [bbox.normalize(self.width, self.height) for bbox in self.bboxes]
        np_arr = np.array([bbox.to_numpy(encode=encode) for bbox in bboxes])
        if include_meta:
            metadata = [[self.img_name, self.width, self.height]] * len(bboxes)
            np_meta = np.array(metadata)
            if encode:
                np_meta = np.char.encode(np_meta, "UTF-8")
            np_arr = np.concatenate([np_arr, np_meta], axis=1)
        return np_arr

    def group_bboxes(
        self, groups: Optional[List[List[int]]] = None, detect_lines=False, **kwargs
    ):
        """Returns a new DetectionResults by merging the bboxes
        If groups is not provided, uses DocTr's default grouping to form them
        len(groups) == number of groups to form
        Each group i.e groups[i] is a list of indexes
        """
        if len(self.bboxes) == 0:
            return self.new()

        if groups is None:
            if detect_lines:
                doc_builder = DocumentBuilder(export_as_straight_boxes=True, **kwargs)
                npy_dets = self.to_numpy(normalize=True)
                npy_bboxes = npy_dets[:, :4].astype(np.float32)
                groups = doc_builder._resolve_lines(npy_bboxes)
            else:
                groups = [range(len(self.bboxes))]

        new_bboxes = [sum(self.bboxes[idx] for idx in group) for group in groups]
        return DetectionResults(new_bboxes, self.width, self.height, self.img_name)

    def filter_by_region(self, x1: float, y1: float, x2: float, y2: float, thresh=0.95):
        """Returns a new DetectionResults
        with only the bboxes within the region provided
        The region is provided in normalized coordinates
        as percentages of the image width and height
        """
        if len(self.bboxes) == 0:
            return self.new()
        x1 = int(x1 * self.width)
        y1 = int(y1 * self.height)
        x2 = int(x2 * self.width)
        y2 = int(y2 * self.height)
        region_bbox = BBox(x1, y1, x2, y2, normalized=False)
        return self.filter_by_bbox(region_bbox, thresh=thresh)

    def expand_bboxes(
        self, up: float = 0, down: float = 0, left: float = 0, right: float = 0
    ):
        """Returns a new DetectionResults
        with the bboxes expanded by up, down, left, right
        """

        if len(self.bboxes) == 0:
            return self.new()
        return DetectionResults(
            [
                bbox.expand(up * bbox.h, down * bbox.h, left * bbox.w, right * bbox.w)
                for bbox in self.bboxes
            ],
            self.width,
            self.height,
            self.img_name,
        )

    def sample(self, k: int):
        """Returns a new DetectionResults
        with k random bboxes.
        """
        samples, _ = get_samples(self.bboxes, k)
        return DetectionResults(samples, self.width, self.height, self.img_name)

    def filter_by_idxs(self, idxs: list):
        """Returns a new DetectionResults
        with only the bboxes at the indexes provided
        """
        if len(self.bboxes) == 0:
            return self.new()

        return DetectionResults(
            [self.bboxes[idx] for idx in idxs], self.width, self.height, self.img_name
        )

    def filter_by_bbox_dist(self, bbox: BBox, p=2, num_keep=2):
        """Returns a new DetectionResults
        with only the closest num_keep bboxes to the provided bbox
        Use the l-p distance metric.
        Example: if p=1, use manhattan, p=2, use euclidean etc.
        """
        dists = [bbox.dist(bbox2, p=p) for bbox2 in self.bboxes]
        idxs = np.argsort(dists)[:num_keep]
        return self.filter_by_idxs(idxs)

    def filter_by_labels(self, labels: list, only_max_conf=False, split=True):
        """Returns a new DetectionResults
        with only the bboxes belonging to the list of labels provided
        If only_max_conf is True, only the bbox with the highest confidence
        is returned for each label
        If split is True, returns a list of DetectionResults for each label
        """
        if len(self.bboxes) == 0:
            return self.new()

        valid_boxes = [bbox for bbox in self.bboxes if bbox.label in labels]
        neg_inf = -9999999
        if only_max_conf:
            dict_max_conf = {}
            for label in labels:
                dict_max_conf[label] = max(
                    [neg_inf]
                    + [bbox.conf for bbox in valid_boxes if bbox.label == label]
                )
            valid_boxes = [
                bbox for bbox in valid_boxes if bbox.conf == dict_max_conf[bbox.label]
            ]
        if split:
            return [
                DetectionResults(
                    [bbox for bbox in valid_boxes if bbox.label == label],
                    self.width,
                    self.height,
                    self.img_name,
                )
                for label in labels
            ]

        return DetectionResults(valid_boxes, self.width, self.height, self.img_name)

    def filter_by_conf(self, conf):
        """Returns a new DetectionResults
        with only the bboxes with the given confidence
        """
        return DetectionResults(
            [bbox for bbox in self.bboxes if bbox.conf >= conf],
            self.width,
            self.height,
            self.img_name,
        )

    def filter_by_bbox(self, bbox: BBox, thresh=0.7, inside=True):
        """Returns a new DetectionResults
        with only the bboxes that intersect with
        the other bboxes with a threshhold >= thresh
        Assumes bbox is denormalized
        If inside is False, returns the bboxes that
        do not intersect with the other bboxes.
        """
        return DetectionResults(
            [b for b in self.bboxes if not (inside ^ b.is_inside(bbox, thresh))],
            self.width,
            self.height,
            self.img_name,
        )

    def filter_by_bbox_text(self, text, strict=False):
        """Returns a new DetectionResults
        with only the bboxes whose texts match the
        given text. String matching is done in lowercase
        Assumes bboxes are TextBBoxes
        """
        if len(self.bboxes) == 0:
            return self.new()

        def match(a, b, strict):
            if strict:
                return a.lower() == b.lower()
            return a.lower() in b.lower()

        return DetectionResults(
            [b for b in self.bboxes if match(text, b.text, strict)],
            self.width,
            self.height,
            self.img_name,
        )

    def new(self, bboxes=None):
        """Returns an empty DetectionResults like self"""
        if bboxes is None:
            bboxes = []
        return DetectionResults(bboxes, self.width, self.height, self.img_name)

    def create_ds(self, parent_ds: "BaseDS"):
        """Creates an ImageDS from the crops of the DetectionResults"""
        source_img = parent_ds[self.img_name]
        crops = self.get_crops(np.array(source_img))
        names = [
            "__".join([get_uuid(), box.label, self.img_name]) for box in self.bboxes
        ]
        image_ds = ImageDS(
            items=crops,
            names=names,
            size=None,
            apply_gs=False,
            batched=False,
        )
        return image_ds

    def get_crops(self, source_img: np.ndarray):
        """Returns a list of crops from the DetectionResults
        Takes each bbox and crops the image
        Returns a list of numpy arrays
        Assumes bboxes are denormalized
        """
        return [
            source_img[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2] for bbox in self.bboxes
        ]

    def sort_bboxes_lr(self):
        """Sorts the bboxes left to right by x coordinate"""
        self.bboxes.sort(key=lambda x: x.x1)
        return self

    def draw(
        self,
        parent_ds: "BaseDS",
        color: tuple = (255, 0, 0),
        show_conf=False,
        show_label=False,
        show_text=False,
        display=True,
    ) -> np.ndarray:
        """Displays the bboxes on a canvas
        If show_conf is True, displays confidence value
        If show_label is True, displays label
        If boxes are normalized, it is denormalized before drawing
        If bbox color is dark, text color is light and vice versa
        """
        canvas = np.array(parent_ds[self.img_name])
        black, white = (0, 0, 0), (255, 255, 255)
        text_color = white if np.mean(color) < 128 else black

        for bbox in self.bboxes:

            labels = []
            if show_label:
                labels.append(bbox.label)
            if show_conf:
                labels.append(f"{bbox.conf:.2f}")
            if show_text:
                labels.append(bbox.text)

            canvas = draw_bbox(
                canvas,
                bbox.values,
                " ".join(labels),
                color=color,
                text_color=text_color,
            )

        if display:
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.imshow(canvas)
            return None
        return canvas
