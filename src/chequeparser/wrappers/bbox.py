import cv2
import numpy as np
from loguru import logger


class BBox:
    def __init__(self, x1, y1, x2, y2, 
                 normalized=True, conf=1.0, label="0"):
        self.x1 = x1 if not normalized else int(x1)
        self.y1 = y1 if not normalized else int(y1)
        self.x2 = x2 if not normalized else int(x2)
        self.y2 = y2 if not normalized else int(y2)
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.cx = (self.x1 + self.x2) / 2
        self.cy = (self.y1 + self.y2) / 2
        self.normalized = normalized
        self.conf = conf
        self.label = label

    @property
    def values(self):
        """Returns coords in the order [x1, y1, x2, y2]"""
        return [self.x1, self.y1, self.x2, self.y2]

    @staticmethod
    def from_xywh(x, y, w, h, normalized=True, conf=1.0, label="0"):
        return BBox(x, y, x + w, y + h, normalized, conf, label)

    @staticmethod
    def from_cxcywh(cx, cy, w, h, normalized=True, conf=1.0, label="0"):
        return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 
                normalized, conf, label)

    def denormalize(self, width, height):
        if not self.normalized:
            logger.info("BBox is already denormalized. Skipping.")
            return self
        return BBox(
            self.x1 * width,
            self.y1 * height,
            self.x2 * width,
            self.y2 * height,
            normalized=False,
            conf=self.conf,
            label=self.label
        )

    def normalize(self, width, height):
        if self.normalized:
            logger.info("BBox is already normalized. Skipping.")
            return self
        return BBox(
            self.x1 / width,
            self.y1 / height,
            self.x2 / width,
            self.y2 / height,
            normalized=True,
            conf=self.conf,
            label=self.label
        )

    def __repr__(self):
        rpr = {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
        return str(rpr)
