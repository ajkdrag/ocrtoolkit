from functools import partial
from ultralytics import YOLO, RTDETR
from doctr.models import recognition_predictor


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


class DOCTR_CRNN_VGG16_PRETRAINED(metaclass=BaseArch):
    @staticmethod
    def load(**kwargs):
        return partial(recognition_predictor,
                       arch="crnn_vgg16_bn",
                       pretrained=True,
                        **kwargs)
