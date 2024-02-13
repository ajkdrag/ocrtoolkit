from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from chequeparser.datasets.imageds import ImageDS
from chequeparser.utilities.draw_utils import draw_bbox
from chequeparser.wrappers.bbox import BBox


class DetectionResults:
    """Wrapper for detection results from a single image
    Captures the resulting bbox detections into a BBox object
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

    def filter_by_labels(self, labels:list, only_max_conf=False):
        """Returns a new DetectionResults
        with only the bboxes belonging to the list of labels provided
        If only_max_conf is True, only the bbox with the highest confidence
        is returned for each label
        """
        valid_boxes = [bbox for bbox in self.bboxes if bbox.label in labels]
        if only_max_conf:
            dict_max_conf = {}
            for label in labels:
                dict_max_conf[label] = max(
                    [bbox.conf for bbox in valid_boxes if bbox.label == label]
                )
            valid_boxes = [
                bbox for bbox in valid_boxes 
                if bbox.conf == dict_max_conf[bbox.label]
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

    def filter_by_bbox(self, bbox: BBox, thresh=0.8):
        """Returns a new DetectionResults
        with only the bboxes that intersect with 
        the other bboxes with a threshhold >= thresh
        """
        return DetectionResults(
            [bbox_ for bbox_ in self.bboxes 
             if bbox_.is_inside(bbox, thresh)],
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
        """
        return [
            self.np_img[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2] for bbox in self.bboxes
        ]

    def draw(
        self,
        color: tuple = (255, 255, 255),
        alpha=0.7,
        show_conf=True,
        show_label=True,
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
            denorm_bbox = bbox

            if bbox.normalized:
                denorm_bbox = bbox.denormalize(self.width, self.height)

            str_label = ""
            if show_label:
                str_label += f"{bbox.label}"
            if show_conf:
                str_label += f" {bbox.conf:.2f}"

            canvas = draw_bbox(
                canvas,
                denorm_bbox.values,
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
