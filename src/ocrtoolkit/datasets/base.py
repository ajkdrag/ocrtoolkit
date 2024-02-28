import math
from collections.abc import Iterable
from functools import partial
from typing import Union

import h5py
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from ocrtoolkit.utilities.img_utils import (
    apply_ops,
    tfm_to_3ch,
    tfm_to_gray,
    tfm_to_pil,
    tfm_to_size,
)
from ocrtoolkit.utilities.misc_utils import get_samples


class BaseDS:
    source = None
    items = []
    names = []
    item_keys = {}
    size = None
    batched = False
    apply_gs = True

    def __init__(
        self,
        source=None,
        items=None,
        names=None,
        size=(640, 320),
        apply_gs=True,
        batched=False,
    ):
        self.source = source
        self.items = items
        self.names = names
        self.size = size
        self.batched = batched
        self.apply_gs = apply_gs
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
        nl_names = [self.names[i] for i in indices]
        return self.__class__(
            source=samples,
            names=nl_names,
            size=self.size,
            apply_gs=self.apply_gs,
            batched=batched,
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
        )

    def setup(self):
        if self.items is None:
            self._setup_items()
        if self.names is None:
            self._setup_names()
        assert len(self.names) == len(self.items)
        assert len(set(self.names)) == len(self.names), "Duplicate names"
        self.item_keys = {name: idx for idx, name in enumerate(self.names)}
        self.item_keys.update({idx: idx for idx in range(len(self.items))})

    @staticmethod
    def empty_like(other: "BaseDS"):
        """Returns an empty ds of type other"""
        return other.__class__(
            source=[],
            batched=other.batched,
            apply_gs=other.apply_gs,
            size=other.size,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key: Union[int, str]):
        """Returns item identified by key
        Applies tfms ops to that item
        """
        if isinstance(key, slice):
            raise NotImplementedError("Slicing not supported")
        try:
            if isinstance(key, int) and key < 0:
                key = len(self.items) + key
            item_key = self.item_keys[key]
        except KeyError:
            raise IndexError(f"Key {key} not found")
        return apply_ops(self.items[item_key], self.tfms)

    def get_as_ds(self, key: Union[int, str, slice, Iterable]) -> "BaseDS":
        """Returns new ds with item at index idx if key is integer,
        else returns new ds with item(s) having the name key
        """
        if isinstance(key, slice):
            items = self.items[key]
            names = self.names[key]
        else:
            key = key if isinstance(key, Iterable) else (key,)
            items = [self.items[self.item_keys[k]] for k in key]
            names = [self.names[self.item_keys[k]] for k in key]
        return self.__class__(
            source=self.source,
            items=items,
            names=names,
            batched=self.batched,
            apply_gs=self.apply_gs,
            size=self.size,
        )

    def _setup_items(self):
        self.items = []

    def _setup_names(self):
        self.names = []

    @staticmethod
    def _serialize_items(items):
        return [np.asarray(item.encode()) for item in items]

    @staticmethod
    def _deserialize_items(items):
        return [item.decode("utf-8") for item in items]

    def train_test_split(self, train_size=0.8, test_size=0.2):
        """Returns train_ds, test_ds"""
        assert train_size + test_size <= 1, "train_size + test_size must be <= 1"
        ids_train, ids_test = train_test_split(
            range(len(self)), train_size=train_size, test_size=test_size
        )
        return (
            self.get_as_ds(ids_train),
            self.get_as_ds(ids_test),
        )

    def save(self, path: str):
        items_data = self.__class__._serialize_items(self.items)
        with h5py.File(path, "w") as f:
            group = f.create_group("class_attributes")
            items = f.create_group("items")

            if self.source is not None:
                group.attrs["source"] = self.source
            if self.size is not None:
                group.attrs["size"] = np.array(self.size)
            group.attrs["apply_gs"] = self.apply_gs
            group.attrs["batched"] = self.batched
            group.create_dataset("names", data=np.array(self.names, dtype="S"))
            for idx, item in enumerate(items_data):
                items.create_dataset(f"item_{idx}", data=item)
            logger.info(f"Dataset saved to {path}")

    @classmethod
    def load(cls, path) -> "BaseDS":
        with h5py.File(path, "r") as f:
            group = f["class_attributes"]
            items = f["items"]
            source = group.attrs.get("source", None)
            size = group.attrs.get("size", None)
            if size is not None:
                size = tuple(size[:])
            apply_gs = group.attrs["apply_gs"]
            batched = group.attrs["batched"]
            names_ds = group["names"]
            names = [name.decode("utf-8") for name in names_ds]
            item_keys = sorted(f["items"].keys(), key=lambda x: int(x.split("_")[-1]))
            item_data = [f["items"][key][()] for key in item_keys]
            items = cls._deserialize_items(item_data)

            logger.info(f"Dataset loaded from {path}")
            return cls(
                source=source,
                names=names,
                size=size,
                apply_gs=apply_gs,
                batched=batched,
                items=items,
            )
