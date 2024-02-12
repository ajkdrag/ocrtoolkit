import numpy as np
from typing import Union
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


def apply_ops(img: Image, ops):
    """Applies list of operations to image"""
    for op in ops:
        img = op(img)
    return img
