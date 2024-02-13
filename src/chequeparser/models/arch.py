from functools import partial

from doctr.models import recognition_predictor
from ultralytics import RTDETR, YOLO


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


class DOCTR_PRETRAINED(metaclass=BaseArch):
    @staticmethod
    def load(**kwargs):
        return partial(recognition_predictor, pretrained=True, **kwargs)
