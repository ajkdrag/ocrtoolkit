from typing import List

import cv2
import numpy as np


class BBox:
    """Wrapper for bounding box"""

    def __init__(self, x1, y1, x2, y2, normalized=False, conf=1.0, label="0"):
        self.x1 = x1 if normalized else int(x1)
        self.y1 = y1 if normalized else int(y1)
        self.x2 = x2 if normalized else int(x2)
        self.y2 = y2 if normalized else int(y2)
        self.eps = 0.0001
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.cx = (self.x1 + self.x2) / 2
        self.cy = (self.y1 + self.y2) / 2
        self.normalized = normalized
        self.conf = conf
        self.label = label
        self.area = self.w * self.h
        self.eps_area = self.area if self.area > self.eps else self.eps
        self.values = [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_xywh(cls, x, y, w, h, normalized=False, conf=1.0, label="0"):
        return cls(x, y, x + w, y + h, normalized, conf, label)

    @classmethod
    def from_cxcywh(cls, cx, cy, w, h, normalized=False, conf=1.0, label="0"):
        return cls(
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, normalized, conf, label
        )

    def denormalize(self, width, height):
        if not self.normalized:
            return self
        return self.__class__(
            self.x1 * width,
            self.y1 * height,
            self.x2 * width,
            self.y2 * height,
            normalized=False,
            conf=self.conf,
            label=self.label,
        )

    def normalize(self, width, height):
        if self.normalized:
            return self
        return self.__class__(
            self.x1 / width,
            self.y1 / height,
            self.x2 / width,
            self.y2 / height,
            normalized=True,
            conf=self.conf,
            label=self.label,
        )

    def expand(self, up, down, left, right):
        """Expands the bounding box by up, down, left, right"""
        return self.__class__(
            self.x1 - left,
            self.y1 - up,
            self.x2 + right,
            self.y2 + down,
            normalized=self.normalized,
            conf=self.conf,
            label=self.label,
        )

    def intersection_area(self, other: "BBox"):
        """Returns the area of intersection between two bboxes
        Assumes same normalization states for both boxes
        """
        assert self.normalized == other.normalized, "Normalization is different"
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        inter_area = max(0, x2 - x1 + self.eps) * max(0, y2 - y1 + self.eps)
        return inter_area

    def is_inside(self, other: "BBox", thresh=0.8):
        """Returns True if the BBox is inside the other BBox
        Calculates the intersection area as IA and self area as A
        If IA / A >= thresh, returns True
        Normalization states for bothe boxes should be same
        """
        assert self.normalized == other.normalized, "Normalization is different"
        IA = self.intersection_area(other)
        A = self.eps_area
        return IA / A >= thresh

    def dist(self, other: "BBox", p=2):
        """Returns the l-p distance between two bboxes"""
        assert p > 0, "p should be > 0"
        assert self.normalized == other.normalized, "Normalization is different"
        return ((self.x1 - other.x1) ** p + (self.y1 - other.y1) ** p) ** (1 / p)

    def as_dict(self):
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "conf": self.conf,
            "label": self.label,
        }

    def __add__(self, other):
        """Returns a new BBox
        with the union of self and other
        Assumes same normalization states for both boxes
        Combines the content of both boxes.
        New box's conf is the max of the confs of the two boxes
        """
        if isinstance(other, self.__class__):
            assert self.normalized == other.normalized, "Normalization is different"
            x1 = min(self.x1, other.x1)
            y1 = min(self.y1, other.y1)
            x2 = max(self.x2, other.x2)
            y2 = max(self.y2, other.y2)
            return self.__class__(
                x1,
                y1,
                x2,
                y2,
                normalized=self.normalized,
                conf=max(self.conf, other.conf),
                label=self.label,
            )
        return self

    def __repr__(self):
        return str(self.as_dict())

    __radd__ = __add__
