from abc import ABCMeta, abstractmethod

import numpy as np


class BaseDetector(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, images):
        raise NotImplementedError
