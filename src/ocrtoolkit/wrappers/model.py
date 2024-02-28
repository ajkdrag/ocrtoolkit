from abc import ABCMeta, abstractmethod
from typing import Any, List

import numpy as np
from loguru import logger

from ocrtoolkit.utilities.img_utils import cv2_tfm_to_3ch
from ocrtoolkit.wrappers.detection_results import DetectionResults
from ocrtoolkit.wrappers.recognition_results import RecognitionResults


class BaseModel(metaclass=ABCMeta):
    model: Any
    path: str
    device: str
    valid_kwargs = set()

    def __init__(self, model, path=None, device="cpu", **kwargs):
        self.model = model
        self.path = path
        self.device = device
        logger.info(f"Loaded model from {self.path}, to {self.device}")

    def preprocess(self, images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        return [cv2_tfm_to_3ch(image) for image in images]

    @abstractmethod
    def _predict(self, image: List[np.ndarray], **kwargs):
        raise NotImplementedError

    def predict(self, images: List[np.ndarray], **kwargs):
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in self.valid_kwargs
        }
        return self._predict(images, **filtered_kwargs)


class DetectionModel(BaseModel):
    def predict(self, images: List[np.ndarray], **kwargs) -> List[DetectionResults]:
        return super().predict(images, **kwargs)


class RecognitionModel(BaseModel):
    def predict(self, images: List[np.ndarray], **kwargs) -> List[RecognitionResults]:
        return super().predict(images, **kwargs)
