from pathlib import Path
from typing import List, Union

from chequeparser.datasets.base import BaseDS
from chequeparser.utilities.img_utils import apply_ops
from chequeparser.utilities.io_utils import get_image_files
from PIL import Image


class FileDS(BaseDS):
    """Allows iterating through list of paths
    Loads image from path
    Applies transformations to image
    Can be iterated through like a list
    """

    raw: Union[str, Path, List[str], List[Path]] = None

    items: Union[List[str], List[Path]] = None

    def load(self):
        """If raw is a single file, converts to list
        Checks if str or Path is a file or a directory
        If it's a single file, then converts to list
        If it's a directory, then returns list of image files
        """
        self.tfms = [Image.open, *self.tfms]
        if isinstance(self.raw, (str, Path)):
            if Path(self.raw).is_file():
                return [self.raw]
            elif Path(self.raw).is_dir():
                return get_img_files(self.raw)
            else:
                raise ValueError(f"{self.raw} is not a file or a dir")
        return self.raw

    def __len__(self):
        return len(self.items)

    def __getitem(self, idx):
        """Returns item at index idx
        Applies tfms ops to that item
        """
        item = self.items[idx]
        return apply_ops(item, self.tfms)
