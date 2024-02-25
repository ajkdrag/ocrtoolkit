import io
from typing import Union

import cv2
import numpy as np
from PIL import Image


def tfm_to_pil(img: Union[Image.Image, np.ndarray]):
    """ "Converts image to PIL image"""
    return Image.fromarray(img) if isinstance(img, np.ndarray) else img


def tfm_to_gray(img: Image):
    """Converts image to grayscale"""
    return img.convert("L")


def tfm_to_size(img: Image, size: tuple):
    """Resizes image to size: (w, h)"""
    return img.resize(size)


def tfm_to_3ch(img: Image):
    """Converts image to 3 channel"""
    return img.convert("RGB")


def cv2_tfm_to_3ch(img: np.ndarray):
    """Converts image to 3 channel if gray"""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def pil_to_bytes(img: Image, format="JPEG"):
    """Converts PIL image to bytes"""
    image_buffer = io.BytesIO()
    img.save(image_buffer, format=format)
    return image_buffer.getvalue()


def bytes_to_pil(image_bytes: bytes):
    """Converts bytes to PIL image"""
    return Image.open(io.BytesIO(image_bytes))


def apply_ops(img: Image, ops):
    """Applies list of operations to image"""
    for op in ops:
        img = op(img)
    return img
