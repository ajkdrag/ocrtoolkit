from pathlib import Path
from typing import List, Union

from PIL import Image

from chequeparser.datasets.base import BaseDS
from chequeparser.utilities.img_utils import apply_ops
from chequeparser.utilities.io_utils import get_image_files


class FileDS(BaseDS):
    """Allows iterating through list of paths
    Loads image from path
    Applies transformations to image
    Can be iterated through like a list
    """

    source: Union[str, Path, List[str], List[Path]] = None
    items: Union[List[str], List[Path]] = None

    def setup(self):
        """If source is a single file, converts to list
        Checks if str or Path is a file or a directory
        If it's a single file, then converts to list
        If it's a directory, then returns list of image files
        """
        self.tfms = [Image.open, *self.tfms]

        if self.items is None:
            if isinstance(self.source, (str, Path)):
                if Path(self.source).is_file():
                    self.items = [self.source]
                elif Path(self.source).is_dir():
                    self.items = get_image_files(self.source)
                else:
                    raise ValueError(f"{self.source} is not a file or a dir")
            else:
                self.items = self.source

        if self.l_parent_idx is None:
            self.l_parent_idx = list(range(len(self.items)))

        if self.names is None:
            self.reset_names()

    def reset_names(self):
        self.names = [Path(item).name for item in self.items]
