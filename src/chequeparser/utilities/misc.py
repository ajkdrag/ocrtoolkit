import random
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image


def is_var_file(var):
    """Checks if variable is a file
    Path and str both are accepted
    """
    if isinstance(var, Path) or isinstance(var, str):
        return Path(var).is_file()
    return False


def is_var_dir(var):
    """Checks if variable is a directory
    Path and str both are accepted
    """
    if isinstance(var, Path) or isinstance(var, str):
        return Path(var).is_dir()
    return False


def is_var_single_image(var):
    """Checks if variable is a single image"""
    return isinstance(var, Image.Image) or isinstance(var, np.ndarray)


def is_var_list_images(var):
    """Checks if variable is a list of images"""
    return isinstance(var, list) and all(
        [isinstance(item, Image.Image) or isinstance(item, np.ndarray) for item in var]
    )


def get_samples(l_source: list, k=5) -> list:
    """Get k random samples from list
    Handles case when k exceeds list length and when k is negative.
    """
    if k > len(l_source):
        k = len(l_source)
    if k < 0:
        logger.error("k cannot be negative")
        raise ValueError("k cannot be negative")
    return random.sample(l_source, k)


def filter_list(l_source, func_tgt, func_cond, num_samples=5) -> list:
    """Builds an output list based on target functions and conditions
    Additionally, for items that do not satisfy the conditions, the count
    and a few samples are returned
    """
    l_final = []
    l_samples = []
    for item in l_source:
        if func_tgt(item):
            out = func_cond(item)
            l_final.append(out)
        else:
            l_samples.append(item)
    return l_final, get_samples(l_samples, num_samples)
