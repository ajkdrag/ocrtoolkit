from abc import ABCMeta, abstractmethod

from functools import partial
from chequeparser.utilities.misc import get_samples
from chequeparser.utilities.img_utils import (
    tfm_to_pil,
    tfm_to_size,
    tfm_to_gray,
)


class BaseDS(metaclass=ABCMeta):
    raw = None
    items = None

    def __init__(self, raw, size=(640, 320), apply_gs=True):
        self.raw = raw
        self.size = size
        self.apply_gs = apply_gs
        self.tfms = [
            tfm_to_pil,
            partial(tfm_to_size, size=size) if size else lambda x: x,
            tfm_to_gray if apply_gs else lambda x: x,
        ]
        self.items = self.load()

    def sample(self, k=5):
        """returns a sample DS of size k"""
        samples = get_samples(self.items, k)
        return self.__class__(samples, size=self.size, apply_gs=self.apply_gs)

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
