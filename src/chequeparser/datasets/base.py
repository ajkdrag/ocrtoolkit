import math
from functools import partial
from chequeparser.utilities.misc import get_samples
from chequeparser.utilities.img_utils import (
    tfm_to_pil,
    tfm_to_size,
    tfm_to_gray,
    tfm_to_3ch,
    apply_ops
)


class BaseDS:
    raw = None
    items = None
    parent_ds = None
    l_parent_idx = None

    def __init__(self, raw, items=None, size=(640, 320), 
                 apply_gs=True, batched=False, 
                 parent_ds=None, l_parent_idx=None):
        self.raw = raw
        self.items = items
        self.size = size
        self.apply_gs = apply_gs
        self.batched = batched
        self.parent_ds = parent_ds
        self.l_parent_idx = l_parent_idx
        self.tfms = [
            tfm_to_pil,
            partial(tfm_to_size, size=size) if size else lambda x: x,
            tfm_to_gray if apply_gs else lambda x: x,
            tfm_to_3ch
        ]

        self.setup()

    def sample(self, k=5, batched=True):
        """returns a sample DS of size k"""
        samples, indices = get_samples(self.items, k)
        nl_parent_idx = [self.l_parent_idx[i] for i in indices]
        return self.__class__(samples, 
                              size=self.size, 
                              apply_gs=self.apply_gs,
                              batched=batched,
                              parent_ds=self,
                              l_parent_idx=nl_parent_idx)

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
            parent_ds=self,
            l_parent_idx=self.l_parent_idx[start:end]
        )

    def setup(self):
        if self.l_parent_idx is None:
            self.l_parent_idx = []
        if self.items is None:
            self.items = []

    @staticmethod
    def empty_like(other: 'BaseDS'):
        """Returns an empty ds of type other"""
        return other.__class__(
            items=[],
            l_parent_idx=[],
            parent_ds=other.parent_ds,
            batched=False,
            apply_gs=False,
            size=None
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """Returns item at index idx
        Applies tfms ops to that item
        """
        item = self.items[idx]
        return apply_ops(item, self.tfms)
