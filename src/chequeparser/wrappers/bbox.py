import cv2
import numpy as np
from loguru import logger


class BBox:
    def __init__(self, x1, y1, x2, y2, normalized=True):
        self.x1 = x1 if not normalized else int(x1)
        self.y1 = y1 if not normalized else int(y1)
        self.x2 = x2 if not normalized else int(x2)
        self.y2 = y2 if not normalized else int(y2)
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.cx = (self.x1 + self.x2) / 2
        self.cy = (self.y1 + self.y2) / 2
        self.normalized = normalized
        self.values = [x1, y1, x2, y2]

    @staticmethod
    def from_xywh(x, y, w, h, normalized=True):
        return BBox(x, y, x + w, y + h, normalized)

    @staticmethod
    def from_cxcywh(cx, cy, w, h, normalized=True):
        return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, normalized)

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
        )

    def draw(self, bg: np.ndarray, color: tuple, thickness=1):
        canvas = bg.copy()
        cv2.rectangle(canvas, (self.x1, self.y1), (self.x2, self.y2), color, thickness)

    def __repr__(self):
        rpr = {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
        return str(rpr)
