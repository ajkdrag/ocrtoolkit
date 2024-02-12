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

    raw: Union[Image.Image, np.ndarray, 
               List[Image.Image], List[np.ndarray]] = None
    items: Union[List[Image.Image], List[np.ndarray]] = None

    def setup(self):
        """If raw is a single image, converts to list"""
        if self.items is None:
            if isinstance(self.raw, (Image.Image, np.ndarray)):
                self.items = [self.raw]
            else: self.items = self.raw

        if self.l_parent_idx is None:
            self.l_parent_idx = list(range(len(self.items)))
