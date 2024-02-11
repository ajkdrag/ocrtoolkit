import math
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

    def __init__(self, raw, size=(640, 320), apply_gs=True, batched=False):
        self.raw = raw
        self.size = size
        self.apply_gs = apply_gs
        self.batched = batched
        self.tfms = [
            tfm_to_pil,
            partial(tfm_to_size, size=size) if size else lambda x: x,
            tfm_to_gray if apply_gs else lambda x: x,
        ]
        self.items = self.load()

    def sample(self, k=5, batched=True):
        """returns a sample DS of size k"""
        samples = get_samples(self.items, k)
        return self.__class__(samples, 
                              size=self.size, 
                              apply_gs=self.apply_gs,
                              batched=batched)

    def num_batches(self, bs=4):
        return math.ceil(len(self) / bs)

    def batch(self, bs=4, bs_idx=0):
        """returns a batch DS of size bs
        Returns the batch corresponding to bs_idx
        Handles the case of out of range batches
        Handles negative bs_idx
        Handle the case where last batch is smaller
        """
        bs = min(bs, len(self))
        if bs_idx < 0: bs_idx += self.num_batches(bs)
        start = max(0, bs_idx * bs)
        end = min(len(self), start + bs)
        return self.__class__(
            self.items[start:end],
            size=self.size,
            apply_gs=self.apply_gs,
            batched=True,
        )

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
