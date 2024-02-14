import math
from functools import partial

from chequeparser.utilities.img_utils import (
    apply_ops,
    tfm_to_3ch,
    tfm_to_gray,
    tfm_to_pil,
    tfm_to_size,
)
from chequeparser.utilities.misc import get_samples


class BaseDS:
    source = None
    items = []
    names = []
    parent_ds = None
    l_parent_idx = None
    size = None
    apply_gs = True
    batched = False

    def __init__(
        self,
        source,
        items=None,
        names=None,
        size=(640, 320),
        apply_gs=True,
        batched=False,
        parent_ds=None,
        l_parent_idx=None,
    ):
        self.source = source
        self.items = items
        self.names = names
        self.size = size
        self.apply_gs = apply_gs
        self.batched = batched
        self.parent_ds = parent_ds
        self.l_parent_idx = l_parent_idx
        self.tfms = [
            tfm_to_pil,
            partial(tfm_to_size, size=size) if size else lambda x: x,
            tfm_to_gray if apply_gs else lambda x: x,
            tfm_to_3ch,
        ]

        self.setup()

    def sample(self, k=5, batched=True):
        """returns a sample DS of size k"""
        samples, indices = get_samples(self.items, k)
        nl_parent_idx = [self.l_parent_idx[i] for i in indices]
        nl_names = [self.names[i] for i in indices]
        return self.__class__(
            source=samples,
            names=nl_names,
            size=self.size,
            apply_gs=self.apply_gs,
            batched=batched,
            parent_ds=self,
            l_parent_idx=nl_parent_idx,
        )

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
        if bs_idx < 0:
            bs_idx += self.num_batches(bs)
        start = max(0, bs_idx * bs)
        end = min(len(self), start + bs)
        return self.__class__(
            source=self.items[start:end],
            names=self.names[start:end],
            size=self.size,
            apply_gs=self.apply_gs,
            batched=True,
            parent_ds=self,
            l_parent_idx=self.l_parent_idx[start:end],
        )

    def setup(self):
        if self.l_parent_idx is None:
            self.l_parent_idx = []
        if self.items is None:
            self.items = []
        if self.names is None:
            self.reset_names()

    @staticmethod
    def empty_like(other: "BaseDS"):
        """Returns an empty ds of type other"""
        return other.__class__(
            source=[],
            l_parent_idx=other.l_parent_idx,
            parent_ds=other.parent_ds,
            batched=other.batched,
            apply_gs=other.apply_gs,
            size=other.size,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """Returns item at index idx
        Applies tfms ops to that item
        """
        item = self.items[idx]
        return apply_ops(item, self.tfms)

    def get_as_ds(self, key: Union[int, str]) -> "BaseDS":
        """Returns new ds with item at index idx if key is integer,
        else returns new ds with item(s) having the name key
        """
        if isinstance(key, int):
            return self.__class__(
                source=[self.items[key]],
                names=[self.names[key]],
                l_parent_idx=[self.l_parent_idx[key]],
                parent_ds=self.parent_ds,
                batched=self.batched,
                apply_gs=self.apply_gs,
                size=self.size,
            )
        else:
            return self.__class__(
                source=[
                    self.items[i] for i in range(len(self)) if self.names[i] == key
                ],
                names=[self.names[i] for i in range(len(self)) if self.names[i] == key],
                l_parent_idx=[
                    self.l_parent_idx[i]
                    for i in range(len(self))
                    if self.names[i] == key
                ],
                parent_ds=self.parent_ds,
                batched=self.batched,
                apply_gs=self.apply_gs,
                size=self.size,
            )


def reset_names(self):
    self.names = []
