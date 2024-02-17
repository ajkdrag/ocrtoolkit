from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from chequeparser.datasets.imageds import ImageDS
from chequeparser.utilities.draw_utils import draw_bbox
from chequeparser.wrappers.bbox import BBox
from chequeparser.wrappers.textbbox import TextBBox


class DetectionResults:
    """Wrapper for detection results from a single image
    Captures the resulting bbox detections into a BBox object
    Assumes all bboxes are denormalized
    Stores the original input image width and height
    Methods to draw the bbox on a canvas
    """

    def __init__(
        self, bboxes: List[BBox], np_img: np.ndarray, parent_ds=None, parent_idx=None
    ):
        self.bboxes = bboxes
        self.width = np_img.shape[1]
        self.height = np_img.shape[0]
        self.np_img = np_img
        self.parent_ds = parent_ds
        self.parent_idx = parent_idx

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        return self.bboxes[idx]

    def to_numpy(self, normalize=False) -> np.ndarray:
        """Returns bboxes as a numpy array
        Each bbox object is converted to a numpy array
        If normalize is True, the bboxes are normalized
        """
        if normalize:
            return np.array(
                [bbox.normalize(self.width, self.height).values for bbox in self.bboxes]
            )
        return np.array([bbox.values for bbox in self.bboxes])

    def group_boxes(self, groups: List[List[int]]):
        """Returns a new DetectionResults by merging the bboxes
        len(groups) == number of groups to form
        Each group i.e groups[i] is a list of indexes
        """
        if len(self.bboxes) == 0:
            return self.empty()

        new_bboxes = [sum([self.bboxes[idx] for idx in group]) for group in groups]
        return DetectionResults(
            new_bboxes, self.np_img, self.parent_ds, self.parent_idx
        )

    def filter_by_region(self, x1: float, y1: float, x2: float, y2: float, thresh=0.95):
        """Returns a new DetectionResults
        with only the bboxes within the region provided
        The region is provided in normalized coordinates
        as percentages of the image width and height
        """
        if len(self.bboxes) == 0:
            return self.empty()
        x1 = int(x1 * self.width)
        y1 = int(y1 * self.height)
        x2 = int(x2 * self.width)
        y2 = int(y2 * self.height)
        region_bbox = BBox(x1, y1, x2, y2, normalized=False)
        return self.filter_by_bbox(region_bbox, thresh=thresh)

    def filter_by_x_range(self, x1: float, x2: float):
        """Returns a new DetectionResults
        with only the bboxes within the x range provided
        The ranges are provided in normalized coordinates
        as percentages of the image width
        """
        if len(self.bboxes) == 0:
            return self.empty()
        x1 = int(x1 * self.width)
        x2 = int(x2 * self.width)

        return DetectionResults(
            [bbox for bbox in self.bboxes if bbox.x1 >= x1 and bbox.x2 <= x2],
            self.np_img,
            self.parent_ds,
            self.parent_idx,
        )

    def expand_bboxes(
        self, up: float = 0, down: float = 0, left: float = 0, right: float = 0
    ):
        """Returns a new DetectionResults
        with the bboxes expanded by up, down, left, right
        """

        if len(self.bboxes) == 0:
            return self.empty()
        return DetectionResults(
            [
                bbox.expand(up * bbox.h, down * bbox.h, left * bbox.w, right * bbox.w)
                for bbox in self.bboxes
            ],
            self.np_img,
            self.parent_ds,
            self.parent_idx,
        )

    def filter_by_idxs(self, idxs: list):
        """Returns a new DetectionResults
        with only the bboxes at the indexes provided
        """
        if len(self.bboxes) == 0:
            return self.empty()

        return DetectionResults(
            [self.bboxes[idx] for idx in idxs],
            self.np_img,
            self.parent_ds,
            self.parent_idx,
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
            return self.empty()

        valid_boxes = [bbox for bbox in self.bboxes if bbox.label in labels]
        NEG_INF = -9999999
        if only_max_conf:
            dict_max_conf = {}
            for label in labels:
                dict_max_conf[label] = max(
                    [NEG_INF]
                    + [bbox.conf for bbox in valid_boxes if bbox.label == label]
                )
            valid_boxes = [
                bbox for bbox in valid_boxes if bbox.conf == dict_max_conf[bbox.label]
            ]
        if split:
            return [
                DetectionResults(
                    [bbox for bbox in valid_boxes if bbox.label == label],
                    self.np_img,
                    self.parent_ds,
                    self.parent_idx,
                )
                for label in labels
            ]

        return DetectionResults(
            valid_boxes,
            self.np_img,
            self.parent_ds,
            self.parent_idx,
        )

    def filter_by_conf(self, conf):
        """Returns a new DetectionResults
        with only the bboxes with the given confidence
        """
        return DetectionResults(
            [bbox for bbox in self.bboxes if bbox.conf >= conf],
            self.np_img,
            self.parent_ds,
            self.parent_idx,
        )

    def filter_by_bbox(self, bbox: BBox, thresh=0.7):
        """Returns a new DetectionResults
        with only the bboxes that intersect with
        the other bboxes with a threshhold >= thresh
        Assumes bbox is denormalized
        """
        return DetectionResults(
            [b for b in self.bboxes if b.is_inside(bbox, thresh)],
            self.np_img,
            self.parent_ds,
            self.parent_idx,
        )

    def filter_by_bbox_text(self, text):
        """Returns a new DetectionResults
        with only the bboxes whose texts match the
        given text. String matching is done in lowercase
        Assumes bboxes are TextBBoxes
        """
        if len(self.bboxes) == 0:
            return self.empty()
        if not isinstance(self.bboxes[0], TextBBox):
            raise NotImplementedError("Only TextBBox supports this operations")
        return DetectionResults(
            [b for b in self.bboxes if text.lower() in b.text.lower()],
            self.np_img,
            self.parent_ds,
            self.parent_idx,
        )

    def empty(self):
        """Returns an empty DetectionResults like self"""
        return DetectionResults(
            [],
            self.np_img,
            self.parent_ds,
            self.parent_idx,
        )

    def create_ds(self):
        """Creates an ImageDS from the crops of the DetectionResults"""
        crops = self.get_crops()
        suffix = self.parent_ds.names[self.parent_idx] if self.parent_ds else ""
        names = ["__".join([box.label, suffix]) for box in self.bboxes]
        image_ds = ImageDS(
            source=crops,
            names=names,
            size=None,
            apply_gs=False,
            batched=False,
            parent_ds=self.parent_ds,
            l_parent_idx=[self.parent_idx] * len(crops),
        )
        return image_ds

    def get_crops(self):
        """Returns a list of crops from the DetectionResults
        Takes each bbox and crops the image
        Returns a list of numpy arrays
        Assumes bboxes are denormalized
        """
        return [
            self.np_img[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2] for bbox in self.bboxes
        ]

    def sort_bboxes_lr_(self):
        """Sorts the bboxes left to right by x coordinate"""
        self.bboxes.sort(key=lambda x: x.x1)

    def draw(
        self,
        color: tuple = (255, 0, 0),
        alpha=0.7,
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
        canvas = self.np_img.copy()
        black, white = (0, 0, 0), (255, 255, 255)
        text_color = white if np.mean(color) < 128 else black

        for bbox in self.bboxes:

            str_label = ""
            if show_label:
                str_label += f"{bbox.label}"
            if show_conf:
                str_label += f" {bbox.conf:.2f}"
            if show_text:
                if not isinstance(bbox, TextBBox):
                    raise NotImplementedError("Only TextBBox supports this operations")
                str_label += f" {bbox.text}"

            canvas = draw_bbox(
                canvas,
                bbox.values,
                str_label,
                color=color,
                text_color=text_color,
            )

        overlay = cv2.addWeighted(canvas, alpha, self.np_img, 1 - alpha, gamma=0)
        if display:
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.imshow(overlay)

        return overlay
