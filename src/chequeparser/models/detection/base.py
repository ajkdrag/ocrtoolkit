from abc import ABCMeta, abstractmethod
from typing import List

import cv2
import numpy as np
from loguru import logger

from chequeparser.wrappers.detection_results import DetectionResults


class BaseDetect(metaclass=ABCMeta):
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def preprocess_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.preprocess(image) for image in images]

    @abstractmethod
    def predict(self, image: np.ndarray) -> DetectionResults:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, images: List[np.ndarray]) -> List[DetectionResults]:
        raise NotImplementedError
