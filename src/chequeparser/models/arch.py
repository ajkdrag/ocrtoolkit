from functools import partial

import torch
from doctr.models import crnn_vgg16_bn, detection_predictor, recognition_predictor
from google.oauth2 import service_account
from loguru import logger
from ultralytics import RTDETR, YOLO


def _load_model_state(path, arch, **kwargs):
    is_pretrained = path is None
    pretrained_backbone = kwargs.get("pretrained_backbone", True)
    model = arch(pretrained=is_pretrained, pretrained_backbone=pretrained_backbone)
    if not is_pretrained:
        logger.info(f"Loading model params from {path}")
        params = torch.load(path)
        model.load_state_dict(params)
    return recognition_predictor(arch=model, pretrained=is_pretrained, **kwargs)


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


class DOCTR_DETECT_PRETRAINED(metaclass=BaseArch):
    @staticmethod
    def load(**kwargs):
        return partial(detection_predictor, pretrained=True, **kwargs)


class DOCTR_RECOG_CRNN_VGG16(metaclass=BaseArch):
    @staticmethod
    def load(**kwargs):
        return partial(_load_model_state, arch=crnn_vgg16_bn, **kwargs)


class GCV_DETECT_RECOG(metaclass=BaseArch):
    @staticmethod
    def load(**kwargs):
        return service_account.Credentials.from_service_account_file(**kwargs)
