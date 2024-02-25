import base64
import inspect
import os
import random
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image


def delegates(kwargs={}, keep=False):
    """Decorator: replace `**kwargs` in signature with params from `to`"""

    def _f(f):
        from_f = f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        for name, kw in kwargs.items():
            sigd[name] = inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default={
                    k: v.default
                    for k, v in inspect.signature(kw).parameters.items()
                    if k not in sigd and v.default != inspect.Parameter.empty
                },
            )
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def get_uuid(num_chars=8):
    return base64.urlsafe_b64encode(os.urandom(num_chars)).decode("ascii")


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
    indices = random.sample(range(len(l_source)), k)
    return [l_source[i] for i in indices], indices


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
    return l_final, get_samples(l_samples, num_samples)[0]


def partition_list(l_source, l_sizes):
    """If l_source=[1, 2, 3, 5], l_sizes=[1, 3], return [[1], [2, 3, 5]]
    Assert len(l_source) == sum(l_sizes)
    """
    assert sum(l_sizes) == len(l_source), "Partition sizes do not add up to list length"
    l_partitions = []
    idx = 0
    for size in l_sizes:
        l_partitions.append(l_source[idx : idx + size])
        idx += size
    return l_partitions
