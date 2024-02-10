from abc import ABCMeta, abstractmethod

from chequeparser.utilities.misc import get_samples


class BaseDS(metaclass=ABCMeta):
    raw = None
    items = None

    def __init__(self, raw, size=(640, 320), apply_gs=True):
        self.raw = raw
        self.items = self.load()
        self.size = size
        self.apply_gs = apply_gs
        self.tfms = [
            tfm_to_pil,
            tfm_to_size if size else lambda x: x,
            tfm_to_gray if apply_gs else lambda x: x,
        ]

    def sample(self, k):
        """returns a random sample DS of items"""
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
