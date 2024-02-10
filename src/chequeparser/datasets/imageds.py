from pathlib import Path
from typing import List, Union

import numpy as np
from chequeparser.datasets.base import BaseDS
from chequeparser.utilities.img_utils import apply_ops
from PIL import Image


class ImageDS(BaseDS):
    """Allows iterating through list of images
    Applies transformations to each image
    Can be iterated through like a list
    """

    raw: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]] = None

    items: Union[List[Image.Image], List[np.ndarray]] = None

    def load(self):
        """If raw is a single image, converts to list"""
        if isinstance(self.raw, (Image.Image, np.ndarray)):
            return [self.raw]
        return self.raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """Returns item at index idx
        Applies tfms ops to that item
        """
        item = self.items[idx]
        return apply_ops(item, self.tfms)
