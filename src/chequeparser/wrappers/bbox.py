import cv2
import numpy as np


class BBox:
    def __init__(self, x1, y1, x2, y2, normalized=True, conf=1.0, label="0"):
        self.x1 = x1 if normalized else int(x1)
        self.y1 = y1 if normalized else int(y1)
        self.x2 = x2 if normalized else int(x2)
        self.y2 = y2 if normalized else int(y2)
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.cx = (self.x1 + self.x2) / 2
        self.cy = (self.y1 + self.y2) / 2
        self.normalized = normalized
        self.conf = conf
        self.label = label
        self.area = self.w * self.h

    @property
    def values(self):
        """Returns coords in the order [x1, y1, x2, y2]"""
        return [self.x1, self.y1, self.x2, self.y2]

    @staticmethod
    def from_xywh(x, y, w, h, normalized=True, conf=1.0, label="0"):
        return BBox(x, y, x + w, y + h, normalized, conf, label)

    @staticmethod
    def from_cxcywh(cx, cy, w, h, normalized=True, conf=1.0, label="0"):
        return BBox(
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, normalized, conf, label
        )

    def denormalize(self, width, height):
        if not self.normalized:
            return self
        return BBox(
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
        return BBox(
            self.x1 / width,
            self.y1 / height,
            self.x2 / width,
            self.y2 / height,
            normalized=True,
            conf=self.conf,
            label=self.label,
        )

    def is_inside(self, other: "BBox", thresh=0.8):
        """Returns True if the BBox is inside the other BBox
        Calculates the intersection area as IA and self area as A
        If IA / A >= thresh, returns True
        Computation is done in denormalized coordinates
        """
        bboxA = self.denormalize(self.width, self.height)
        bboxB = other.denormalize(other.width, other.height)
        IA = (min(bboxA.x2, bboxB.x2) - max(bboxA.x1, bboxB.x1)) * (
            min(bboxA.y2, bboxB.y2) - max(bboxA.y1, bboxB.y1)
        )
        A = self.w * self.h
        return IA / A >= thresh

    def __repr__(self):
        rpr = {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
        return str(rpr)
