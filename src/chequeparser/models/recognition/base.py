from abc import ABCMeta, abstractmethod
from typing import List

import cv2
import numpy as np
from loguru import logger

from chequeparser.wrappers.recognition_results import RecognitionResults


class BaseRecognize(metaclass=ABCMeta):
    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def preprocess_batch(self, images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        return [self.preprocess(image) for image in images]

    def predict(self, image: np.ndarray, **kwargs) -> RecognitionResults:
        return self.predict_batch([image], **kwargs)[0]

    @abstractmethod
    def predict_batch(
        self, images: List[np.ndarray], **kwargs
    ) -> List[RecognitionResults]:
        raise NotImplementedError
