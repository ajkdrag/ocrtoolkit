from ast import literal_eval
from typing import Union

import numpy as np

from ocrtoolkit.wrappers.recognition_results import RecognitionResults


class BBox:
    """Wrapper for a bounding box."""

    x1: Union[int, float]  #: x coordinate of the top left corner.
    y1: Union[int, float]  #: y coordinate of the top left corner.
    x2: Union[int, float]  #: x coordinate of the bottom right corner.
    y2: Union[int, float]  #: y coordinate of the bottom right corner.
    normalized: bool  #: Whether the coordinates are normalized.
    conf: float  #: Confidence of the detection.
    label: str  #: Label of the detection.
    text: str  #: Text of the detection.
    text_conf: float  #: Confidence of the text.
    eps: float = 1e-6  #: Epsilon to avoid division by zero.
    w: Union[int, float]  #: Width of the bounding box.
    h: Union[int, float]  #: Height of the bounding box.
    cx: Union[int, float]  #: Center x coordinate of the bounding box.
    cy: Union[int, float]  #: Center y coordinate of the bounding box.
    area: Union[int, float]  #: Area of the bounding box.
    eps_area: Union[int, float]  #: Epsilon if the area is zero else area.
    values: list  #: Coordinates of the bounding box in the format [x1, y1, x2, y2].

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
        """Set the text and confidence of the bounding box ocr result.

        Args:
            text_and_conf (Union[tuple, "RecognitionResults"]): OCR text and confidence.
        """
        if isinstance(text_and_conf, tuple):
            self.text, self.text_conf = text_and_conf
        else:
            self.text, self.text_conf = text_and_conf.text, text_and_conf.conf

    @classmethod
    def from_xywh(
        cls, x, y, w, h, normalized=False, conf=1.0, label="0", text="", text_conf=0
    ):
        """Create a BBox from x, y, w, h coordinates.

        Args:
            x: x coordinate.
            y: y coordinate.
            w: Width of the box.
            h: Height of the box.
            normalized: Whether the coordinates are normalized.
            conf: Confidence of the box.
            label: Label of the box.
            text: OCR text of the box.
            text_conf: OCR text confidence of the box.

        Returns:
            BBox: A BBox object.
        """
        return cls(x, y, x + w, y + h, normalized, conf, label, text, text_conf)

    @classmethod
    def from_cxcywh(
        cls, cx, cy, w, h, normalized=False, conf=1.0, label="0", text="", text_conf=0
    ):
        """Create a BBox from cx, cy, w, h coordinates.

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            w: Width of the box.
            h: Height of the box.
            normalized: Whether the coordinates are normalized.
            conf: Confidence of the box.
            label: Label of the box.
            text: OCR text of the box.
            text_conf: OCR text confidence of the box.

        Returns:
            BBox: A BBox object.
        """
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
        """Denormalize the bounding box according to the width and height of the image.

        Args:
            width: Width of the image.
            height: Height of the image.

        Returns:
            BBox: A denormalized BBox object.
        """
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
        """Normalize the bounding box according to the width and height of the image.

        Args:
            width: Width of the image.
            height: Height of the image.

        Returns:
            BBox: A normalized BBox object.
        """
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
        """Expand the bounding box by up, down, left, right.

        Args:
            up: Amount to expand up.
            down: Amount to expand down.
            left: Amount to expand left.
            right: Amount to expand right.

        Returns:
            BBox: A expanded BBox object.
        """
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
        """Return the area of intersection between two bboxes.

        Assumes same normalization states for both boxes

        Args:
            other: Other BBox.

        Returns:
            float: Area of intersection.
        """
        assert self.normalized == other.normalized, "Normalization is different"
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        inter_area = max(0, x2 - x1 + self.eps) * max(0, y2 - y1 + self.eps)
        return inter_area

    def is_inside(self, other: "BBox", thresh=0.8):
        """Return True if this BBox is inside the other BBox.

        Calculates the intersection area as IA and self area as A.
        If IA / A >= thresh, returns True.
        Normalization states for both boxes should be same.

        Args:
            other: Other BBox.
            thresh: Threshold for IA / A.

        Returns:
            bool: True if this BBox is inside the other BBox with the given threshold.
        """
        assert self.normalized == other.normalized, "Normalization is different"
        IA = self.intersection_area(other)
        A = self.eps_area
        return IA / A >= thresh

    def dist(self, other: "BBox", p=2):
        """Return the l-p distance between two bboxes

        Args:
            other: Other BBox.
            p: The order of the distance. E.g. p=2 for Euclidean distance.

        Returns:
            float: The l-p distance.
        """
        assert p > 0, "p should be > 0"
        assert self.normalized == other.normalized, "Normalization is different"
        return ((self.x1 - other.x1) ** p + (self.y1 - other.y1) ** p) ** (1 / p)

    def apply_text_op(self, op, lowercase=False):
        """Apply an operation to the text of the BBox.

        Args:
            op: Operation to apply to the text.
            lowercase: Whether to lowercase the text.

        Returns:
            BBox: A new BBox object with the operation applied to the text.
        """
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
        """Return a numpy array with all the values of the BBox

        Args:
            encode: Whether to encode the text using UTF-8.

        Returns:
            numpy.ndarray: A numpy array with all the values of the BBox.
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
        """Return a BBox from a numpy array. Casting is done explicitly.

        Args:
            arr: A numpy array.

        Returns:
            BBox: A BBox object.

        """
        arr = np.char.decode(arr.astype(np.bytes_), "UTF-8")
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
        """Return the bbox as a dictionary.

        Returns:
            dict: A dictionary representation of the BBox.
        """
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
        """Return a new BBox with the union of self and other.

        Assumes same normalization states for both boxes.
        New box's text is the concatenation of the texts of the two boxes.
        New box's conf is the max of the confs of the two boxes.

        Args:
            other: Other BBox.

        Returns:
            BBox: A new BBox with the union of self and other.
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
        """Return a string representation of the BBox.

        Returns:
            str: A string representation of the BBox.
        """
        return str(self.to_dict())

    __radd__ = __add__
