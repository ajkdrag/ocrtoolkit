from ast import literal_eval
from typing import Union

import numpy as np

from ocrtoolkit.wrappers.recognition_results import RecognitionResults


class BBox:
    """Wrapper for bounding box"""

    x1: Union[int, float]
    y1: Union[int, float]
    x2: Union[int, float]
    y2: Union[int, float]
    normalized: bool
    conf: float
    label: str
    text: str
    text_conf: float

    def __init__(
        self,
        x1,
        y1,
        x2,
        y2,
        normalized=False,
        conf=1.0,
        label="0",
        text="",
        text_conf=0,
    ):
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
        self.text = text
        self.text_conf = text_conf

    def set_text_and_confidence(
        self, text_and_conf: Union[tuple, "RecognitionResults"]
    ):
        if isinstance(text_and_conf, tuple):
            self.text, self.text_conf = text_and_conf
        else:
            self.text, self.text_conf = text_and_conf.text, text_and_conf.conf

    @classmethod
    def from_xywh(
        cls, x, y, w, h, normalized=False, conf=1.0, label="0", text="", text_conf=0
    ):
        return cls(x, y, x + w, y + h, normalized, conf, label, text, text_conf)

    @classmethod
    def from_cxcywh(
        cls, cx, cy, w, h, normalized=False, conf=1.0, label="0", text="", text_conf=0
    ):
        return cls(
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
            normalized,
            conf,
            label,
            text,
            text_conf,
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
            text=self.text,
            text_conf=self.text_conf,
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
            text=self.text,
            text_conf=self.text_conf,
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
            text=self.text,
            text_conf=self.text_conf,
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

    def apply_text_op(self, op, lowercase=False):
        new_text = self.text.lower() if lowercase else self.text
        new_text = op(new_text)
        return self.__class__(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.normalized,
            self.conf,
            self.label,
            text=new_text,
            text_conf=self.text_conf,
        )

    def to_numpy(self, encode=False):
        """Returns a numpy array
        with all the values of the BBox
        as well as the label and conf
        Also encodes each item to UTF-8
        """
        np_arr = np.array(
            [
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.normalized,
                self.conf,
                self.label,
                self.text,
                self.text_conf,
            ]
        )
        if encode:
            return np.char.encode(np_arr, "UTF-8")
        return np_arr

    @classmethod
    def from_numpy(cls, arr):
        """Returns a BBox from a numpy array
        Casting is done explicitly
        """
        arr = np.char.decode(arr.astype(np.bytes_), "UTF-8")
        # arr = np.array(
        #    [item.decode() if isinstance(item, bytes) else item for item in arr]
        # )
        x1, y1, x2, y2 = map(float, arr[:4])
        normalized = literal_eval(arr[4]) if len(arr) > 4 else False
        conf = float(arr[5]) if len(arr) > 5 else 1.0
        label = str(arr[6]) if len(arr) > 6 else "0"
        text = str(arr[7]) if len(arr) > 7 else ""
        text_conf = float(arr[8]) if len(arr) > 8 else 1.0
        return cls(
            x1,
            y1,
            x2,
            y2,
            normalized,
            conf,
            label,
            text,
            text_conf,
        )

    def to_dict(self):
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "normalized": self.normalized,
            "conf": self.conf,
            "label": self.label,
            "text": self.text,
            "text_conf": self.text_conf,
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
                text=" ".join([self.text, other.text]),
                text_conf=max(self.text_conf, other.text_conf),
            )
        return self

    def __repr__(self):
        return str(self.to_dict())

    __radd__ = __add__
