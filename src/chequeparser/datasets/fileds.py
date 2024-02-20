from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image

from chequeparser.datasets.base import BaseDS
from chequeparser.utilities.img_utils import apply_ops
from chequeparser.utilities.io_utils import get_files


class FileDS(BaseDS):
    """Allows iterating through list of paths
    Loads image from path
    Applies transformations to image
    Can be iterated through like a list
    """

    source: Union[str, Path, List[str], List[Path]] = None
    items: Union[List[str], List[Path]] = None

    def setup(self):
        self.tfms = [Image.open, *self.tfms]
        super().setup()

    def _setup_items(self):
        if isinstance(self.source, (str, Path)):
            if Path(self.source).is_file():
                self.items = [self.source]
            elif Path(self.source).is_dir():
                self.items = get_files(self.source)
            else:
                raise ValueError(f"{self.source} is not a file or a dir")
        else:
            self.items = self.source

    def _setup_names(self):
        self.names = [Path(item).name for item in self.items]

    @staticmethod
    def _serialize_items(items):
        """Converts items to numpy array for serialization"""
        return np.array(items, dtype="S")

    @staticmethod
    def _deserialize_items(np_items):
        """Gets items from numpy array"""
        return [item.decode("utf-8") for item in np_items]
