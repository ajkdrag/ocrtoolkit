from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image

from chequeparser.datasets.base import BaseDS
from chequeparser.utilities.img_utils import (
    apply_ops,
    bytes_to_pil,
    pil_to_bytes,
    tfm_to_pil,
)


class ImageDS(BaseDS):
    """Allows iterating through list of images
    Applies transformations to each image
    Can be iterated through like a list
    """

    source: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]] = None
    items: Union[List[Image.Image], List[np.ndarray]] = None

    def _setup_items(self):
        """If source is a single image, converts to list"""
        if isinstance(self.source, (Image.Image, np.ndarray)):
            self.items = [self.source]
        else:
            self.items = self.source

    def _setup_names(self):
        self.names = [f"Image: {i}" for i in range(len(self.items))]

    @staticmethod
    def _serialize_items(items):
        return [np.asarray(pil_to_bytes(tfm_to_pil(item))) for item in items]

    @staticmethod
    def _deserialize_items(items):
        return [bytes_to_pil(item) for item in items]
