from functools import partial
from ultralytics import YOLO, RTDETR


class BaseArch(type):
    def __call__(cls, **kwargs):
        return cls.load(**kwargs)


class UL_YOLO(metaclass=BaseArch):
    @staticmethod
    def load(**kwargs):
        return partial(YOLO, **kwargs)

class UL_RTDETR(metaclass=BaseArch):
    @staticmethod
    def load(**kwargs):
        return partial(RTDETR, **kwargs)
