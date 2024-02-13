from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image

from chequeparser.datasets.base import BaseDS
from chequeparser.utilities.img_utils import apply_ops


class ImageDS(BaseDS):
    """Allows iterating through list of images
    Applies transformations to each image
    Can be iterated through like a list
    """

    source: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]] = None
    items: Union[List[Image.Image], List[np.ndarray]] = None

    def setup(self):
        """If source is a single image, converts to list"""
        if self.items is None:
            if isinstance(self.source, (Image.Image, np.ndarray)):
                self.items = [self.source]
            else:
                self.items = self.source

        if self.l_parent_idx is None:
            self.l_parent_idx = list(range(len(self.items)))

        if self.names is None:
            self.reset_names()

    def reset_names(self):
        self.names = [f"Image: {i}" for i in range(len(self.items))]
